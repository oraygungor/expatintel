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
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Global Kayıt Kilidi (Concurrency güvenliği için)
save_lock = asyncio.Lock()

# Eşzamanlı işlem sınırı
MAX_CONCURRENT_SCRAPES = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)

# Kritik Kelimeler ve Kaynak Puanları
EXPAT_KEYWORDS = ["skat", "tax", "visa", "permit", "residence", "immigration", "work", "job", "salary", "rent", "housing", "bolig", "integration", "turkish", "turkey", "student", "regeringen", "law", "kommune"]
TRUSTED_SOURCES = ["dr.dk", "politiken.dk", "berlingske.dk", "thelocal.dk"]

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
    """HTML etiketlerini temizler, lxml yoksa html.parser kullanır."""
    if not raw_html: return ""
    try:
        return BeautifulSoup(raw_html, "lxml").get_text(separator=" ", strip=True)
    except:
        return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)

def canonicalize_url(url):
    """URL'yi normalize eder ve takip parametrelerini temizler."""
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
    """Çekilen metnin kalitesini kontrol eder (Boilerplate tespiti)."""
    if not text or len(text) < 400: return False
    # Çerez uyarısı veya menü metni baskınlığı kontrolü
    bad_tokens = ["cookie", "privacy policy", "accept all", "subscribe", "newsletter", "terms of service", "all rights reserved"]
    bad_count = sum(1 for token in bad_tokens if token in text.lower())
    # Eğer metnin çok küçük bir kısmında bu kelimeler çok fazlaysa kalitesiz kabul edebiliriz
    if bad_count > 5: return False
    return True

def parse_date_to_utc(entry):
    """RSS tarihini UTC aware datetime nesnesine çevirir."""
    for attr in ['published_parsed', 'updated_parsed', 'created_parsed']:
        if hasattr(entry, attr) and getattr(entry, attr):
            # struct_time'ı UTC kabul ederek çeviriyoruz
            return datetime(*getattr(entry, attr)[:6], tzinfo=timezone.utc)
    return None

def get_soft_score(title, source):
    """Haber için kural tabanlı bir ön puan üretir."""
    score = 0
    title_lower = title.lower()
    if any(src in source.lower() for src in TRUSTED_SOURCES): score += 2
    if any(kw in title_lower for kw in EXPAT_KEYWORDS): score += 3
    if "denmark" in title_lower or "danish" in title_lower: score += 1
    return score

def get_recent_news():
    """RSS kaynaklarından haberleri toplar ve ön filtreleme yapar."""
    raw_articles = []
    seen_urls = set()
    now_utc = datetime.now(timezone.utc)
    time_limit = now_utc - timedelta(hours=12)
    
    logger.info("1. AŞAMA: RSS Taraması Başlatıldı (UTC).")
    
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        source = feed.feed.get("title", urlparse(url).netloc)
        
        for entry in feed.entries:
            pub_date = parse_date_to_utc(entry)
            is_recent = pub_date and pub_date >= time_limit
            
            link = canonicalize_url(entry.get("link", ""))
            if not link or link in seen_urls: continue
            
            title = entry.get("title", "").strip()
            summary = clean_html(entry.get("summary", ""))[:400]
            
            soft_score = get_soft_score(title, source)
            
            article_data = {
                "id": len(raw_articles),
                "title": title,
                "link": link,
                "source": source,
                "summary": summary,
                "published_at": pub_date.isoformat() if pub_date else None,
                "soft_score": soft_score,
                "is_recent": is_recent
            }
            raw_articles.append(article_data)
            seen_urls.add(link)
            
    return raw_articles

async def score_news_with_llm(client, news_list):
    """LLM ile ID tabanlı ve kesin şemalı puanlama yapar."""
    # Sadece yeni olan veya kural tabanlı puanı yüksek olanları gönder
    candidates = [n for n in news_list if n['is_recent'] or n['soft_score'] >= 3]
    if not candidates: return []
    
    logger.info(f"2. AŞAMA: {len(candidates)} haber LLM puanlamasına gönderiliyor.")
    
    context = ""
    for n in candidates:
        context += f"ID: {n['id']} | Başlık: {n['title']} | Özet: {n['summary']}\n\n"

    prompt = f"""
    Danimarka haberlerini expat ilgisi açısından (1-10) analiz et.
    Sadece puanı 8 ve üzeri olanları seç.
    
    KESİN JSON ŞEMASI:
    {{
      "selected_news": [
        {{ "id": 0, "expat_score": 9, "reasoning": "kısa gerekçe" }}
      ]
    }}
    
    Haberler:
    {context}
    """

    for attempt in range(3): 
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={ "type": "json_object" },
                messages=[{"role": "system", "content": "Profesyonel haber küratörü."}, {"role": "user", "content": prompt}]
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
            
            logger.info(f"LLM {len(final_selection)} haber seçti.")
            return final_selection
        except Exception as e:
            logger.warning(f"Puanlama hatası (Deneme {attempt+1}): {e}")
            await asyncio.sleep(2 ** attempt)
    return []

async def fetch_url_content(url):
    """İçerik çeker ve kalite kontrolü yapar."""
    async with semaphore:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        
        # Deneme hiyerarşisi
        methods = ["trafilatura", "playwright", "firecrawl"]
        for method in methods:
            try:
                content = None
                if method == "trafilatura":
                    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as c:
                        res = await c.get(url, headers=headers)
                        content = trafilatura.extract(res.text)
                
                elif method == "playwright":
                    from playwright.async_api import async_playwright
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(headless=True)
                        page = await browser.new_page(user_agent=headers["User-Agent"])
                        await page.goto(url, wait_until="networkidle", timeout=30000)
                        content = trafilatura.extract(await page.content())
                        await browser.close()

                elif method == "firecrawl" and FIRECRAWL_API_KEY:
                    async with httpx.AsyncClient(timeout=30.0) as c:
                        response = await c.post("https://api.firecrawl.dev/v1/scrape", 
                            headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                            json={"url": url, "formats": ["markdown"]})
                        content = response.json().get("data", {}).get("markdown") if response.status_code == 200 else None

                if is_quality_content(content):
                    return content, method
            except:
                continue
        return None, "failed"

async def analyze_and_save(client, article, existing_urls):
    """Haberin tam analizini yapar ve atomik olarak kaydeder."""
    if article['link'] in existing_urls: return

    content, method = await fetch_url_content(article['link'])
    if not content: return

    prompt = f"""
    Haber metnini Türk expatlar için analiz et. 
    KESİN JSON ŞEMASI:
    {{
      "translated_title": "Tarafsız Türkçe Başlık",
      "summary": "2 cümlelik özet",
      "expat_impact": "2 cümlelik etki analizi"
    }}
    
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
        
        await atomic_save(final_data)
        logger.info(f"[+] Kaydedildi: {article['title'][:40]}...")
    except Exception as e:
        logger.error(f"Analiz hatası: {e}")

async def atomic_save(new_entry, filename="daily_news.json"):
    """Global kilit kullanarak atomik kayıt yapar."""
    async with save_lock:
        data = {"haberler": []}
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try: data = json.load(f)
                except: pass
        
        # Tekrar kontrolü
        if any(h['link'] == new_entry['link'] for h in data['haberler']):
            return

        data['haberler'].insert(0, new_entry)
        
        # Güvenli dosya değişimi
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(filename)))
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as tmp:
                json.dump(data, tmp, ensure_ascii=False, indent=2)
            os.replace(temp_path, filename)
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            logger.error(f"Dosya kayıt hatası: {e}")

async def main():
    if not API_KEY:
        logger.error("OPENAI_API_KEY eksik!")
        return

    client = AsyncOpenAI(api_key=API_KEY)
    
    # 1. RSS Verilerini Topla
    raw_news = get_recent_news()
    
    # 2. LLM Puanlaması (ID Tabanlı)
    selected_news = await score_news_with_llm(client, raw_news)
    
    if not selected_news:
        logger.info("İşlenecek yeni ve uygun haber bulunamadı.")
        return

    # 3. Mevcut Verileri Oku
    filename = "daily_news.json"
    existing_urls = set()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try: 
                d = json.load(f)
                existing_urls = {h['link'] for h in d.get('haberler', [])}
            except: pass

    # 4. Paralel Analiz ve Kayıt
    tasks = [analyze_and_save(client, article, existing_urls) for article in selected_news]
    await asyncio.gather(*tasks)
    
    logger.info("Bot çalışması tamamlandı.")

if __name__ == "__main__":
    asyncio.run(main())
