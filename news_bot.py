import json
import feedparser
import time
import os
import asyncio
import httpx
import trafilatura
import logging
import tempfile
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from openai import AsyncOpenAI

# --- YAPILANDIRMA VE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Global Kayıt Kilidi
save_lock = asyncio.Lock()

# Eşzamanlı işlem sınırı
MAX_CONCURRENT_SCRAPES = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)

# Kritik Kelimeler ve Kaynak Puanları
EXPAT_KEYWORDS = ["skat", "tax", "visa", "permit", "residence", "immigration", "work", "job", "salary", "rent", "housing", "bolig", "integration", "turkish", "turkey", "student", "regeringen", "law", "kommune"]
TRUSTED_SOURCES = ["dr.dk", "politiken.dk", "berlingske.dk", "thelocal.dk", "ritzau.dk"]

RSS_FEEDS = [
    "https://feeds.thelocal.com/rss/dk",
    "https://news.google.com/rss/search?q=Denmark+when:12h&hl=en-US&gl=US&ceid=US:en",
    "https://www.dr.dk/nyheder/service/feeds/allenyheder",
    "https://politiken.dk/rss/senestenyt.rss",
    "https://www.berlingske.dk/content/rss",
    "https://www.information.dk/feed",
    "https://www.reddit.com/r/Denmark/.rss",
    "https://via.ritzau.dk/rss/short-messages/latest"
]

def resolve_url(url: str) -> str:
    if "news.google.com" not in url:
        return url
    try:
        from googlenewsdecoder import gnewsdecoder
        result = gnewsdecoder(url, interval=1)
        if isinstance(result, dict):
            for key in ("decoded_url", "url", "original_url"):
                val = result.get(key)
                if val and val.startswith("http"): return val
        return result if isinstance(result, str) and result.startswith("http") else url
    except Exception as e:
        return url

def clean_html(raw_html):
    if not raw_html: return ""
    try:
        return BeautifulSoup(raw_html, "lxml").get_text(separator=" ", strip=True)
    except:
        return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)

def canonicalize_url(url):
    try:
        u = urlparse(url)
        query = parse_qs(u.query)
        blacklisted = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'igsh', 'feature'}
        filtered_query = {k: v for k, v in query.items() if k.lower() not in blacklisted}
        new_query = urlencode(filtered_query, doseq=True)
        return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment)).lower().rstrip('/')
    except:
        return url.lower()

def is_quality_content(text):
    if not text: 
        return False, "İçerik boş"
    
    clean_text = text.strip()
    if len(clean_text) < 400: 
        return False, f"Metin çok kısa ({len(clean_text)} karakter)"
    
    bad_tokens = ["cookie", "privacy policy", "accept all", "subscribe", "newsletter", "terms of service"]
    bad_count = sum(1 for token in bad_tokens if token in clean_text.lower())
    
    if bad_count > 5: 
        return False, f"Çok fazla çöp/reklam kelimesi ({bad_count})"
    
    return True, "Geçerli içerik"

def parse_date_to_utc(entry):
    for attr in ['published_parsed', 'updated_parsed', 'created_parsed']:
        if hasattr(entry, attr) and getattr(entry, attr):
            return datetime(*getattr(entry, attr)[:6], tzinfo=timezone.utc)
    return None

def get_soft_score(title, source):
    score = 0
    title_lower = title.lower()
    if any(src in source.lower() for src in TRUSTED_SOURCES): score += 2
    if any(kw in title_lower for kw in EXPAT_KEYWORDS): score += 3
    return score

def get_recent_news():
    raw_articles = []
    seen_urls = set()
    now_utc = datetime.now(timezone.utc)
    # KESİN FİLTRE: Sadece son 48 saat (Bugün ve Dün)
    time_limit = now_utc - timedelta(hours=48)
    
    print(f"\n[1/4] RSS TARAMASI BAŞLATILDI (Zaman Sınırı: {time_limit.strftime('%d.%m.%Y %H:%M')})...")
    
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        source = feed.feed.get("title", urlparse(url).netloc)
        found_on_feed = 0
        
        for entry in feed.entries:
            pub_date = parse_date_to_utc(entry)
            
            # Tarih filtresi (Tarihi olmayan veya eski olan haberler anında elenir)
            if not pub_date or pub_date < time_limit:
                continue
            
            raw_link = entry.get("link", "")
            if not raw_link: continue
            
            resolved_link = resolve_url(raw_link)
            link = canonicalize_url(resolved_link)
            
            if link in seen_urls: continue
            
            title = entry.get("title", "").strip()
            summary = clean_html(entry.get("summary", ""))[:400]
            soft_score = get_soft_score(title, source)
            
            article_data = {
                "id": len(raw_articles),
                "title": title,
                "link": link,
                "source": source,
                "summary": summary,
                "published_at": pub_date.isoformat(),
                "soft_score": soft_score,
                "is_recent": True
            }
            raw_articles.append(article_data)
            seen_urls.add(link)
            found_on_feed += 1
            
        print(f" -> {source[:30]:<30}: {found_on_feed} yeni haber bulundu.")
            
    return raw_articles

async def score_news_with_llm(client, news_list):
    # SADECE güncel haberler LLM'e gider (is_recent garantisi ile)
    candidates = [n for n in news_list if n['soft_score'] >= 1]
    if not candidates: return []
    
    print(f"\n[2/4] LLM PUANLAMASI: {len(candidates)} güncel aday haber analiz ediliyor...")
    
    context = ""
    for n in candidates:
        context += f"ID: {n['id']} | Başlık: {n['title']} | Özet: {n['summary']}\n\n"

    prompt = """
    Danimarka haberlerini expat ilgisi (1-10) açısından analiz et. Sadece 8+ olanları seç.
    JSON ŞEMASI: { "selected_news": [ { "id": 0, "expat_score": 9, "reasoning": "..." } ] }
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[{"role": "system", "content": "Haber küratörü."}, {"role": "user", "content": f"{prompt}\n\nHaberler:\n{context}"}]
        )
        result = json.loads(response.choices[0].message.content)
        selection_map = {item['id']: item for item in result.get("selected_news", [])}
        
        final_selection = []
        for n in candidates:
            if n['id'] in selection_map:
                n.update({
                    "expat_score": selection_map[n['id']].get("expat_score"),
                    "selection_reason": selection_map[n['id']].get("reasoning")
                })
                final_selection.append(n)
        
        print(f" -> LLM {len(final_selection)} adet yüksek öncelikli haber belirledi.")
        return final_selection
    except Exception as e:
        logger.error(f"Puanlama hatası: {e}")
        return []

async def fetch_url_content(url, article_id):
    async with semaphore:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
        prefix = f"    [Haber #{article_id}]"

        # --- 1. Yöntem: Trafilatura ---
        print(f"{prefix} [Yöntem 1] Trafilatura deneniyor: {url}")
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as c:
                res = await c.get(url, headers=headers)
                content = trafilatura.extract(res.text)
                is_ok, reason = is_quality_content(content)
                if is_ok:
                    print(f"{prefix}       [+] BAŞARILI: Trafilatura ile {len(content)} karakter çekildi.")
                    return content, "trafilatura"
                else:
                    print(f"{prefix}       [-] Atlandı: {reason}")
        except Exception as e:
            print(f"{prefix}       [!] Hata: {type(e).__name__}")

        # --- 2. Yöntem: Playwright ---
        print(f"{prefix} [Yöntem 2] Playwright (JS) deneniyor: {url}")
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(user_agent=headers["User-Agent"])
                await page.goto(url, wait_until="networkidle", timeout=30000)
                html = await page.content()
                await browser.close()
                
                content = trafilatura.extract(html)
                is_ok, reason = is_quality_content(content)
                if is_ok:
                    print(f"{prefix}       [+] BAŞARILI: Playwright ile {len(content)} karakter çekildi.")
                    return content, "playwright"
                else:
                    print(f"{prefix}       [-] Kalite Yetersiz: {reason}")
        except Exception as e:
            print(f"{prefix}       [!] Playwright Hatası: {str(e)[:100]}")

        # --- 3. Yöntem: Firecrawl ---
        if FIRECRAWL_API_KEY:
            print(f"{prefix} [Yöntem 3] Firecrawl API deneniyor: {url}")
            try:
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    response = await http_client.post(
                        "https://api.firecrawl.dev/v1/scrape", 
                        headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                        json={"url": url, "formats": ["markdown"]}
                    )
                    if response.status_code == 200:
                        content = response.json().get("data", {}).get("markdown")
                        is_ok, reason = is_quality_content(content)
                        if is_ok:
                            print(f"{prefix}       [+] BAŞARILI: Firecrawl ile {len(content)} karakter çekildi.")
                            return content, "firecrawl"
                        else:
                            print(f"{prefix}       [-] Firecrawl Kalite Engeli: {reason}")
                    else:
                        print(f"{prefix}       [-] Firecrawl API Hatası ({response.status_code})")
            except Exception as e:
                print(f"{prefix}       [!] Firecrawl Hata: {type(e).__name__}")
        
        return None, "failed"

async def analyze_and_save(client, article, existing_urls, file_name):
    if article['link'] in existing_urls:
        return

    print(f"\n--- HABER İŞLENİYOR [ID:{article['id']}]: {article['title'][:60]}... ---")
    print(f"    [Link]: {article['link']}")
    
    content, method = await fetch_url_content(article['link'], article['id'])
    if not content:
        print(f"    [ID:{article['id']}] [X] ATLANDI: İçerik çekilemedi.")
        return

    print(f"    [ID:{article['id']}] [LLM] İçerik analiz ediliyor...")
    prompt = f"""
    Metni Türk expatlar için analiz et. 
    JSON: {{ "translated_title": "...", "summary": "...", "expat_impact": "..." }}
    Metin: {content[:6000]}
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[{"role": "user", "content": prompt}]
        )
        analysis = json.loads(response.choices[0].message.content)
        
        final_data = {
            **article,
            **analysis,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "scrape_method": method,
            "content_length": len(content)
        }
        
        async with save_lock:
            await atomic_save(final_data, file_name)
        print(f"    [ID:{article['id']}] [V] TAMAMLANDI: Arşive eklendi.")
    except Exception as e:
        logger.error(f"Analiz hatası (ID:{article['id']}): {e}")

async def atomic_save(new_entry, filename):
    data = {"haberler": []}
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try: data = json.load(f)
            except: pass
    
    if any(h['link'] == new_entry['link'] for h in data['haberler']):
        return

    data['haberler'].insert(0, new_entry)
    
    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(filename)))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
        os.replace(temp_path, filename)
    except:
        if os.path.exists(temp_path): os.remove(temp_path)

async def main():
    if not API_KEY:
        print("HATA: OPENAI_API_KEY bulunamadı!")
        return

    client = AsyncOpenAI(api_key=API_KEY)
    file_name = "daily_news.json"
    
    # 1. RSS Verilerini Topla
    raw_news = get_recent_news()
    
    # 2. LLM Puanlaması
    selected_news = await score_news_with_llm(client, raw_news)
    
    if not selected_news:
        print("\nSonuç: Bugün işlenecek kritik yeni bir haber bulunamadı.")
        return

    existing_urls = set()
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            try: 
                d = json.load(f)
                existing_urls = {h['link'] for h in d.get('haberler', [])}
            except: pass

    print(f"\n[3/4] İÇERİK ÇEKME VE ANALİZ ({len(selected_news)} Haber, Max 3 Paralel)...")
    tasks = [analyze_and_save(client, article, existing_urls, file_name) for article in selected_news]
    await asyncio.gather(*tasks)
    
    print(f"\n[4/4] İŞLEM TAMAMLANDI. 'daily_news.json' güncellendi.\n")

if __name__ == "__main__":
    asyncio.run(main())
