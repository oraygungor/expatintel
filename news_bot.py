import json
import feedparser
import time
import os
import asyncio
import httpx
import trafilatura
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import AsyncOpenAI

# --- GÜVENLİK ---
API_KEY = os.getenv("OPENAI_API_KEY")

RSS_FEEDS = [
    "https://feeds.thelocal.com/rss/dk",
    "https://news.google.com/rss/search?q=Denmark+when:12h&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Denmark+when:12h&hl=da&gl=DK&ceid=DK:da",
    "https://www.dr.dk/nyheder/service/feeds/allenyheder",
    "https://politiken.dk/rss/senestenyt.rss",
    "https://www.berlingske.dk/content/rss",
    "https://www.information.dk/feed",
    "https://fyens.dk/feed/danmark",
    "https://nordjyske.dk/rss/nyheder",
    "https://jv.dk/feed/danmark",
    "https://hsfo.dk/feed/danmark",
    "https://frdb.dk/feed/danmark",
    "https://www.reddit.com/r/Denmark/.rss"
]

def resolve_url(url: str) -> str:
    """Google News vb. yönlendirme linklerini çözer."""
    if "news.google.com" not in url and "reddit.com" not in url:
        return url
    try:
        from googlenewsdecoder import gnewsdecoder
        if "news.google.com" in url:
            result = gnewsdecoder(url, interval=1)
            if isinstance(result, dict):
                for key in ("decoded_url", "url", "original_url"):
                    val = result.get(key)
                    if val and val.startswith("http"): return val
            return result if isinstance(result, str) and result.startswith("http") else url
        return url
    except ImportError:
        return url
    except Exception:
        return url

def get_recent_news():
    """RSS kaynaklarından son 12 saatteki haberleri çeker."""
    articles = []
    seen_links = set()
    now = datetime.now()
    time_limit = now - timedelta(hours=12) # SON 12 SAAT
    
    print(f"\n--- 1. AŞAMA: RSS TARAMASI BAŞLADI ({now.strftime('%H:%M:%S')}) ---")
    
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        source_name = feed.feed.get("title", url.split('/')[2])
        feed_total, new_on_feed = 0, 0
        
        for entry in feed.entries:
            feed_total += 1
            pub_date = None
            
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                pub_date = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
            
            # Tarih yoksa veya son 12 saat içindeyse
            if not pub_date or pub_date >= time_limit:
                final_link = resolve_url(entry.get("link", ""))
                if not final_link or final_link in seen_links: continue

                title = entry.get("title", "").strip()
                summary = entry.get("summary", "")[:500]
                
                seen_links.add(final_link)
                new_on_feed += 1
                
                articles.append({
                    "title": title,
                    "link": final_link,
                    "source": source_name,
                    "summary": summary,
                    "date": now.strftime("%d.%m.%Y")
                })
        
        print(f"[{source_name[:30]:<30}] Taranan: {feed_total:<3} | Uygun (Son 12s): {new_on_feed}")
    
    return articles

async def score_news_with_llm(client, news_list):
    """Tüm haberleri LLM'e gönderip 1-10 arası puanlatır (8 ve üzeri olanları seçer)."""
    if not news_list: return []
    
    print(f"\n--- 2. AŞAMA: {len(news_list)} HABER LLM İLE PUANLANIYOR ---")
    
    context = ""
    for idx, item in enumerate(news_list):
        context += f"ID: {idx}\nBaşlık: {item['title']}\nÖzet: {item['summary']}\nLink: {item['link']}\n\n"

    prompt = f"""
    Aşağıdaki Danimarka haberlerini Danimarka'da yaşayan Türk expatları (beyaz yakalılar, öğrenciler, çalışanlar) ilgilendirme derecesine göre 1 ile 10 arasında puanla.
    Yasal düzenlemeler, göçmenlik, vergiler, çalışma hayatı, Danimarka-Türkiye ilişkileri gibi konular yüksek puan almalıdır.
    
    SADECE puanı 8 ve üzeri olan haberlerin linklerini ve expat_puani değerini JSON olarak döndür.
    
    JSON FORMATI:
    {{
      "secilen_linkler": [
        {{
          "link": "https://orneklink.com/haber1",
          "expat_puani": 9
        }}
      ]
    }}
    
    HABER HAVUZU:
    {context}
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-5.4-mini", 
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "Sen profesyonel bir haber analistisin. Sadece JSON dönersin."},
                {"role": "user", "content": prompt}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        secilenler = result.get("secilen_linkler", [])
        
        valid_links = []
        for s in secilenler:
            try:
                if int(s.get("expat_puani", 0)) >= 8:
                    valid_links.append(s.get("link"))
            except ValueError:
                pass
                
        print(f"Puanlama bitti. 8 ve üzeri puan alan haber sayısı: {len(valid_links)}")
        return valid_links
    except Exception as e:
        print(f"Puanlama Hatası: {e}")
        return []

async def fetch_url_content(url):
    """httpx ile siteye bağlanır, trafilatura ve bs4 ile ana metni çeker."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.google.com/"
    }
    
    try:
        async with httpx.AsyncClient(follow_redirects=True, verify=False) as http_client:
            response = await http_client.get(url, headers=headers, timeout=20.0)
            response.raise_for_status()
            html = response.text
            
            # 1. Yöntem: Trafilatura (Menü, reklam ve yorumları temizler)
            text = trafilatura.extract(html, include_tables=False, include_comments=False)
            
            # 2. Yöntem: BeautifulSoup (Trafilatura başarısız olursa)
            if not text or len(text) < 50:
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)[:5000]
                
            return text if text else ""
            
    except Exception as e:
        print(f"İçerik çekilemedi ({url}): {e}")
        return ""

async def analyze_full_article(client, text, link):
    """Haberin tam metnini LLM'e atıp başlık, özet ve etki analizi üretir."""
    # LLM'i çok yormamak için metni makul bir uzunlukta kesiyoruz
    text = text[:8000] 
    
    prompt = f"""
    Aşağıda Danimarka basınından alınmış bir haberin tam metni bulunmaktadır.
    Bu metni okuyarak aşağıdaki üç şeyi üret:
    1. Haberin içeriğini en doğru şekilde yansıtan, clickbait'ten uzak, tamamen tarafsız yepyeni bir TÜRKÇE BAŞLIK.
    2. Haberin tamamen tarafsız, sade bir dille yazılmış, net ve anlaşılır 2 CÜMLELİK ÖZETİ.
    3. Bu gelişmenin Danimarka'da yaşayan Türk expatlara (beyaz yakalılar, öğrenciler, vb.) olası etkilerinin 2 CÜMLELİK ANALİZİ.
    
    Haber Metni:
    {text}
    
    SADECE aşağıdaki JSON formatında yanıt ver:
    {{
      "yeni_baslik": "Tarafsız ve içeriği yansıtan yeni Türkçe başlık",
      "ozet": "2 cümlelik tarafsız özet...",
      "ai_analiz": "Türk expatlara 2 cümlelik potansiyel etki analizi..."
    }}
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-5.4-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "Sen stratejik bir haber editörü ve sosyo-ekonomik analistsin. Sadece JSON dönersin."},
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Analiz Hatası ({link}): {e}")
        return None

def get_existing_links(file_name="daily_news.json"):
    """JSON'daki mevcut linkleri getirir."""
    if not os.path.exists(file_name):
        return set()
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {h.get("link") for h in data.get("haberler", [])}
    except Exception:
        return set()

def save_single_news(news_data, file_name="daily_news.json"):
    """Tek bir haberi anında JSON'a ekler."""
    if os.path.exists(file_name):
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = {"haberler": []}
    else:
        data = {"haberler": []}
        
    data["haberler"].insert(0, news_data) # En başa ekle
    
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

async def main():
    if not API_KEY:
        print("HATA: OPENAI_API_KEY ortam değişkeni bulunamadı!")
        return

    client = AsyncOpenAI(api_key=API_KEY)
    file_name = "daily_news.json"
    
    # Adım 1: RSS'ten Son 12 Saatlik Haberleri Çek
    raw_news = get_recent_news()
    
    # Adım 2: Toplu Puanlama (1-10 Arası, Sadece >= 8)
    high_score_links = await score_news_with_llm(client, raw_news)
    
    # Adım 3: Mükerrer Kontrolü (Daha önce işlenmiş mi?)
    existing_links = get_existing_links(file_name)
    new_links_to_process = [link for link in high_score_links if link not in existing_links]
    
    if not new_links_to_process:
        print("\nSonuç: İşlenecek yeni, yüksek puanlı bir haber bulunamadı.")
        return
        
    print(f"\n--- 3. AŞAMA: {len(new_links_to_process)} YENİ HABERİN İÇERİĞİ ÇEKİLİP ANALİZ EDİLİYOR ---")
    
    for link in new_links_to_process:
        print(f"\nİşleniyor: {link}")
        
        # Adım 4: Haberin Tam İçeriğini Çek (httpx + trafilatura/bs4)
        full_text = await fetch_url_content(link)
        
        if len(full_text) < 100:
            print(" -> Yeterli metin bulunamadı, bu haber atlanıyor.")
            continue
            
        # Adım 5: Haberi LLM'de Analiz Et
        analysis_result = await analyze_full_article(client, full_text, link)
        
        if analysis_result:
            final_news_object = {
                "baslik": analysis_result.get("yeni_baslik", "Başlık Bulunamadı"),
                "ozet": analysis_result.get("ozet", ""),
                "ai_analiz": analysis_result.get("ai_analiz", ""),
                "link": link,
                "tarih": datetime.now().strftime("%d.%m.%Y")
            }
            
            # Adım 6: Kaydet
            save_single_news(final_news_object, file_name)
            print(f" -> BAŞARIYLA EKLENDİ: {final_news_object['baslik']}")

if __name__ == "__main__":
    # Async döngüyü başlat
    asyncio.run(main())
