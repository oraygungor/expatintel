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
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from openai import AsyncOpenAI

# --- YAPILANDIRMA VE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Eşzamanlı işlem sınırı (Concurrency limit)
MAX_CONCURRENT_SCRAPES = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)

# Ön filtreleme anahtar kelimeleri (Maliyet optimizasyonu için)
EXPAT_KEYWORDS = [
    "skat", "tax", "visa", "permit", "residence", "immigration", "work", "job", 
    "salary", "rent", "housing", "bolig", "integration", "turkish", "turkey", 
    "tyrkiet", "student", "university", "regeringen", "law", "lovforslag"
]

RSS_FEEDS = [
    "https://feeds.thelocal.com/rss/dk",
    "https://news.google.com/rss/search?q=Denmark+when:12h&hl=en-US&gl=US&ceid=US:en",
    "https://www.dr.dk/nyheder/service/feeds/allenyheder",
    "https://politiken.dk/rss/senestenyt.rss",
    "https://www.berlingske.dk/content/rss",
    "https://www.information.dk/feed",
    "https://www.reddit.com/r/Denmark/.rss"
]

def clean_html(raw_html):
    """HTML etiketlerini temizler."""
    if not raw_html: return ""
    return BeautifulSoup(raw_html, "lxml").get_text(separator=" ", strip=True)

def canonicalize_url(url):
    """URL'yi normalize eder (tracking parametrelerini siler)."""
    try:
        u = urlparse(url)
        query = parse_qs(u.query)
        # Sadece önemli olmayan parametreleri temizle
        blacklisted_params = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'igsh'}
        filtered_query = {k: v for k, v in query.items() if k.lower() not in blacklisted_params}
        
        new_query = urlencode(filtered_query, doseq=True)
        return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment)).lower().rstrip('/')
    except:
        return url.lower()

def parse_date(entry):
    """RSS entry'sinden tarih ayıklar, başarısız olursa None döner."""
    for attr in ['published_parsed', 'updated_parsed', 'created_parsed']:
        if hasattr(entry, attr) and getattr(entry, attr):
            return datetime.fromtimestamp(time.mktime(getattr(entry, attr)))
    return None

def get_recent_news():
    """RSS kaynaklarından haberleri toplar ve ID'ler ile haritalar."""
    raw_articles = []
    seen_urls = set()
    now = datetime.now()
    time_limit = now - timedelta(hours=12)
    
    logger.info("1. AŞAMA: RSS Taraması Başlatıldı.")
    
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        source = feed.feed.get("title", urlparse(url).netloc)
        
        for entry in feed.entries:
            pub_date = parse_date(entry)
            is_recent = pub_date and pub_date >= time_limit
            
            link = canonicalize_url(entry.get("link", ""))
            if not link or link in seen_urls: continue
            
            title = entry.get("title", "").strip()
            summary = clean_html(entry.get("summary", ""))[:400]
            
            # Keyword Ön Filtreleme
            full_text_to_check = (title + " " + summary).lower()
            has_keyword = any(kw in full_text_to_check for kw in EXPAT_KEYWORDS)
            
            article_data = {
                "id": len(raw_articles),
                "title": title,
                "link": link,
                "source": source,
                "summary": summary,
                "published_at": pub_date.isoformat() if pub_date else None,
                "is_recent": is_recent,
                "has_keyword": has_keyword
            }
            raw_articles.append(article_data)
            seen_urls.add(link)
            
    logger.info(f"Toplam {len(raw_articles)} aday haber bulundu.")
    return raw_articles

async def score_news_with_llm(client, news_list):
    """LLM ile ID tabanlı puanlama yapar."""
    candidates = [n for n in news_list if n['is_recent'] or n['has_keyword']]
    if not candidates: return []
    
    logger.info(f"2. AŞAMA: {len(candidates)} haber LLM puanlamasına gönderiliyor.")
    
    context = ""
    for n in candidates:
        context += f"ID: {n['id']} | Başlık: {n['title']} | Özet: {n['summary']}\n\n"

    prompt = f"""
    Aşağıdaki Danimarka haberlerini expat (göçmen) ilgisi açısından (1-10) puanla.
    Sadece puanı 8 ve üzeri olanların ID'lerini JSON listesi olarak dön.
    Format: {{"selected_ids": [0, 5, 12], "reasoning": "kısa açıklama"}}
    
    Haberler:
    {context}
    """

    for attempt in range(3): 
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={ "type": "json_object" },
                messages=[{"role": "system", "content": "Sen bir haber analistisin."}, {"role": "user", "content": prompt}]
            )
            result = json.loads(response.choices[0].message.content)
            selected_ids = result.get("selected_ids", [])
            logger.info(f"LLM {len(selected_ids)} haber seçti.")
            return [n for n in candidates if n['id'] in selected_ids]
        except Exception as e:
            logger.warning(f"Puanlama hatası (Deneme {attempt+1}): {e}")
            await asyncio.sleep(2 ** attempt)
    return []

async def fetch_url_content(url):
    """Haber metnini çeker (Trafilatura -> Playwright -> Firecrawl)."""
    async with semaphore:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        
        # 1. Yöntem: Trafilatura (Hızlı)
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as c:
                res = await c.get(url, headers=headers)
                res.raise_for_status()
                text = trafilatura.extract(res.text)
                if text and len(text) > 400: return text, "trafilatura"
        except Exception as e:
            logger.debug(f"Trafilatura başarısız: {e}")

        # 2. Yöntem: Playwright (JS gerektiren siteler için)
        try:
            logger.info(f"Playwright deneniyor: {url[:40]}...")
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(user_agent=headers["User-Agent"])
                # Sayfanın yüklenmesini bekle
                await page.goto(url, wait_until="networkidle", timeout=30000)
                content = await page.content()
                await browser.close()
                text = trafilatura.extract(content)
                if text and len(text) > 400: return text, "playwright"
        except Exception as e:
            logger.warning(f"Playwright başarısız: {e}")

        # 3. Yöntem: Firecrawl (Son çare)
        if FIRECRAWL_API_KEY:
            try:
                logger.info(f"Firecrawl deneniyor: {url[:40]}...")
                async with httpx.AsyncClient(timeout=30.0) as c:
                    response = await c.post(
                        "https://api.firecrawl.dev/v1/scrape",
                        headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                        json={"url": url, "formats": ["markdown"]}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        text = data.get("data", {}).get("markdown", "")
                        if text and len(text) > 400: return text, "firecrawl"
            except Exception as e:
                logger.debug(f"Firecrawl başarısız: {e}")
                
        return None, "failed"

async def analyze_and_save(client, article, existing_urls):
    """Haber analizini yapar ve kaydeder."""
    if article['link'] in existing_urls: return

    content, method = await fetch_url_content(article['link'])
    if not content: return

    prompt = f"Bu haberi Türk expatlar için analiz et. Başlık, 2 cümlelik özet ve etki yorumu üret. Sadece JSON dön.\nMetin: {content[:6000]}"
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[{"role": "user", "content": prompt}]
        )
        analysis = json.loads(response.choices[0].message.content)
        
        final_data = {
            **article,
            "translated_title": analysis.get("yeni_baslik") or analysis.get("baslik"),
            "ai_summary": analysis.get("ozet"),
            "expat_impact": analysis.get("ai_analiz") or analysis.get("analiz"),
            "fetched_at": datetime.now().isoformat(),
            "scrape_method": method
        }
        
        await atomic_save(final_data)
        logger.info(f"[+] Kaydedildi: {article['title'][:40]}...")
    except Exception as e:
        logger.error(f"Analiz hatası: {e}")

async def atomic_save(new_entry, filename="daily_news.json"):
    """JSON dosyasına atomik kayıt."""
    async with asyncio.Lock():
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
    client = AsyncOpenAI(api_key=API_KEY)
    logger.info(f"Sistem Hazır. Firecrawl: {'OK' if FIRECRAWL_API_KEY else 'Eksik'}")
    
    raw_news = get_recent_news()
    selected_news = await score_news_with_llm(client, raw_news)
    
    if not selected_news:
        logger.info("İşlenecek yeni haber bulunamadı.")
        return

    filename = "daily_news.json"
    existing_urls = set()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try: 
                d = json.load(f)
                existing_urls = {h['link'] for h in d.get('haberler', [])}
            except: pass

    tasks = [analyze_and_save(client, article, existing_urls) for article in selected_news]
    await asyncio.gather(*tasks)
    logger.info("İşlem başarıyla tamamlandı.")

if __name__ == "__main__":
    asyncio.run(main())
