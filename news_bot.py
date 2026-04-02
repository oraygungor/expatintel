import json
import feedparser
import time
import os
import re
from datetime import datetime, timedelta
from openai import OpenAI
from googlenewsdecoder import gnewsdecoder

# --- GÜVENLİK ---
# GitHub Secrets üzerinden "OPENAI_API_KEY" adıyla tanımladığın anahtarı çeker.
API_KEY = os.getenv("OPENAI_API_KEY")

RSS_FEEDS = [
    # Global & İngilizce Danimarka Haberleri
    "https://feeds.thelocal.com/rss/dk",
    "https://news.google.com/rss/search?q=Denmark+when:1d&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Denmark+when:1d&hl=da&gl=DK&ceid=DK:da",
    
    # Ulusal Danimarka Kaynakları (Danca)
    "https://www.dr.dk/nyheder/service/feeds/allenyheder",
    "https://politiken.dk/rss/senestenyt.rss",
    "https://www.berlingske.dk/content/rss",
    "https://www.information.dk/feed",
    
    # Bölgesel Danimarka Kaynakları (Danca)
    "https://fyens.dk/feed/danmark",
    "https://nordjyske.dk/rss/nyheder",
    "https://jv.dk/feed/danmark",
    "https://hsfo.dk/feed/danmark",
    "https://frdb.dk/feed/danmark",
    
    # Sosyal Medya & Forum (Reddit)
    "https://www.reddit.com/r/Denmark/.rss"
]

def resolve_url(url: str) -> str:
    """Google News ve Reddit linklerini asıl haber sitesine çevirir."""
    if "news.google.com" not in url and "reddit.com" not in url:
        return url
    try:
        if "news.google.com" in url:
            result = gnewsdecoder(url, interval=1)
            if isinstance(result, dict):
                for key in ("decoded_url", "url", "original_url"):
                    val = result.get(key)
                    if val and val.startswith("http"): return val
            return result if isinstance(result, str) and result.startswith("http") else url
        return url
    except:
        return url

def get_recent_news():
    articles = []
    seen_links = set()
    seen_titles = set()
    now = datetime.now()
    time_limit = now - timedelta(hours=24) # Sadece son 24 saat
    
    print(f"\n--- HABER TARAMA BAŞLADI ({now.strftime('%H:%M:%S')}) ---")
    
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        source_name = feed.feed.get("title", url.split('/')[2])
        
        feed_total = 0
        new_on_feed = 0
        
        for entry in feed.entries:
            feed_total += 1
            pub_date = None
            
            # Tarih verisini farklı RSS formatlarına göre ayıkla
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                pub_date = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
            
            if not pub_date or pub_date >= time_limit:
                title = entry.get("title", "").strip()
                if not title or title.lower() in seen_titles: continue
                
                final_link = resolve_url(entry.get("link", ""))
                if final_link in seen_links: continue

                seen_links.add(final_link)
                seen_titles.add(title.lower())
                new_on_feed += 1
                
                articles.append({
                    "title": title,
                    "link": final_link,
                    "date": pub_date.strftime("%d.%m.%Y") if pub_date else now.strftime("%d.%m.%Y"),
                    "source": source_name,
                    "summary": entry.get("summary", "")[:350]
                })
        
        print(f"[{source_name[:35]:<35}] Taranan: {feed_total:<3} | Yeni: {new_on_feed}")
    
    return articles

def filter_with_ai(news_list):
    if not news_list: return {"haberler": []}
    if not API_KEY:
        print("HATA: OpenAI API Anahtarı bulunamadı!")
        return {"haberler": []}
        
    client = OpenAI(api_key=API_KEY)
    context = ""
    for idx, item in enumerate(news_list):
        context += f"ID: {idx}\nKaynak: {item['source']}\nBaşlık: {item['title']}\nTarih: {item['date']}\nLink: {item['link']}\n\n"

    # Talep ettiğin model versiyonu talimatı eklendi
    prompt = f"""
    Aşağıdaki haberlerden Danimarka'daki expat'ları, hatta özellikle Türk expatları ilgilendirenleri seç. 
    
    ÖNEMLİ KURALLAR:
    1. Sonuçları TÜRKÇE olarak dön.
    2. 'tarih' alanına KESİNLİKLE sadece GG.AA.YYYY formatında tarih yaz (Örn: 09.03.2026).
    3. ASLA "tarih mevcut değil" gibi cümleler kurma. Eğer tarih yoksa bugünün tarihini ({datetime.now().strftime("%d.%m.%Y")}) kullan.
    4. Puanı 8 ve üzeri olanları seç.

    JSON FORMATI:
    {{
      "haberler": [
        {{
          "baslik": "Türkçe Başlık",
          "kaynak": "Kaynak",
          "link": "Link",
          "tarih": "09.03.2026",
          "sebep": "Neden önemli",
          "expat_puani": 9
        }}
      ]
    }}
    
    HABER HAVUZU:
    {context}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-5.4", 
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "Sen profesyonel bir haber analiz uzmanısın. Sadece JSON dönersin."},
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"AI Analiz Hatası: {e}")
        return {"haberler": []}

def safe_date_parse(date_str):
    """Tarih hatalarını önlemek için güvenli parser."""
    try:
        return datetime.strptime(date_str, "%d.%m.%Y")
    except:
        return datetime(2000, 1, 1)

def save_and_merge(new_data):
    file_name = "daily_news.json"
    
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                data = {"haberler": []}
    else:
        data = {"haberler": []}

    existing_links = {h["link"] for h in data.get("haberler", [])}
    added_count = 0
    
    for h in new_data.get("haberler", []):
        if h["link"] not in existing_links:
            if "tarih" not in h or not h["tarih"] or len(h["tarih"]) < 8:
                h["tarih"] = datetime.now().strftime("%d.%m.%Y")
            data["haberler"].append(h)
            added_count += 1
    
    # Tüm arşivi tarihe göre yeniden sırala
    data["haberler"].sort(key=lambda x: safe_date_parse(x.get("tarih", "")), reverse=True)

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n--- İŞLEM BAŞARIYLA TAMAMLANDI ---")
    print(f"Yeni Haber Sayısı: {added_count} | Toplam Arşiv: {len(data['haberler'])}")

if __name__ == "__main__":
    raw_news = get_recent_news()
    if raw_news:
        print(f"\n{len(raw_news)} benzersiz haber AI'ya gönderiliyor...")
        ai_output = filter_with_ai(raw_news)
        save_and_merge(ai_output)
    else:
        print("\nSon 24 saatte yeni içerik bulunamadı.")
