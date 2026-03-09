import json
import feedparser
import time
from datetime import datetime, timedelta
from openai import OpenAI
from googlenewsdecoder import gnewsdecoder
import os

# GitHub Secrets üzerinden alınacak
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RSS_FEEDS = [
    "https://feeds.thelocal.com/rss/dk",
    "https://www.dr.dk/nyheder/service/feeds/allenyheder",
    "https://nyheder.tv2.dk/rss",
    "https://politiken.dk/rss/senestenyt.rss",
    "https://www.berlingske.dk/content/rss",
    "https://www.information.dk/feed",
    "https://nordjyske.dk/rss/nyheder",
    "https://news.google.com/rss/search?q=Denmark+when:7d&hl=en-US&gl=US&ceid=US:en"
]

def resolve_url(url: str) -> str:
    """Google News linklerini çözer, diğerlerini olduğu gibi bırakır."""
    if "news.google.com" not in url:
        return url
    try:
        result = gnewsdecoder(url, interval=1)
        if isinstance(result, dict):
            for key in ("decoded_url", "url", "source_url", "original_url"):
                value = result.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    return value
        return url
    except:
        return url

def get_recent_news():
    articles = []
    seen_links = set()
    seen_titles = set()
    now = datetime.now()
    time_limit = now - timedelta(hours=24)
    
    print(f"Haberler taranıyor... ({len(RSS_FEEDS)} kaynak)")
    
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        source_name = feed.feed.get("title", url.split('/')[2])
        
        for entry in feed.entries:
            # Tarih kontrolü
            pub_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            
            # Son 24 saat filtresi
            if not pub_date or pub_date >= time_limit:
                title = entry.get("title", "").strip()
                date_str = pub_date.strftime("%d.%m.%Y") if pub_date else now.strftime("%d.%m.%Y")
                
                # Başlık bazlı mükerrer kontrolü
                if title.lower() in seen_titles: continue
                
                # Link çözme ve link bazlı mükerrer kontrolü
                final_link = resolve_url(entry.get("link", ""))
                if final_link in seen_links: continue

                seen_links.add(final_link)
                seen_titles.add(title.lower())
                
                articles.append({
                    "title": title,
                    "link": final_link,
                    "date": date_str,
                    "source": source_name,
                    "summary": entry.get("summary", "")[:300] # GPT'ye ipucu olması için kısa özet
                })
    return articles

def filter_with_ai(news_list):
    if not news_list: return {"haberler": []}
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    context = ""
    for idx, item in enumerate(news_list):
        context += f"ID: {idx}\nKaynak: {item['source']}\nBaşlık: {item['title']}\nÖzet: {item['summary']}\nLink: {item['link']}\n\n"

    prompt = f"""
    Sen Danimarka'da yaşayan expat'lar için profesyonel bir haber editörüsün. 
    Aşağıdaki haber havuzunda hem İngilizce hem de Danca (Danish) haberler bulunmaktadır.
    
    GÖREVİN:
    1. SADECE Danimarka'daki expat'ları (yabancı çalışanlar, öğrenciler, oturum izni sahipleri) ilgilendiren haberleri seç.
    2. Seçtiğin haberlerin başlıklarını ve 'sebep' kısmını TÜRKÇE olarak yaz.
    3. Puanı 8 ve üzeri olan en kritik haberleri belirle.
    4. Aynı olayı anlatan haberlerden sadece 1 tanesini (en kaliteli kaynağı) seç.
    5. Tarih formatı GG.AA.YYYY olmalıdır.

    ÇIKTI FORMATI (JSON):
    {{
      "haberler": [
        {{
          "baslik": "Türkçe Başlık",
          "kaynak": "Kaynak Adı",
          "link": "Haber Linki",
          "tarih": "GG.AA.YYYY",
          "sebep": "Neden önemli (Türkçe)",
          "expat_puani": 9
        }}
      ]
    }}
    
    Haber Havuzu:
    {context}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # gpt-5.4 henüz yayınlanmadığı için stabil gpt-4o kullanılmıştır.
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "Sen sadece yapılandırılmış JSON çıktı veren bir haber analiz uzmanısın."},
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"GPT Hatası: {e}")
        return {"haberler": []}

def save_and_merge(new_data):
    file_name = "daily_news.json"
    
    # Mevcut dosyayı oku
    if os.path.exists(file_name):
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except:
            existing_data = {"haberler": []}
    else:
        existing_data = {"haberler": []}

    existing_links = {h["link"] for h in existing_data.get("haberler", [])}
    
    # Yenileri ekle
    added_count = 0
    for h in new_data.get("haberler", []):
        if h["link"] not in existing_links:
            existing_data["haberler"].append(h)
            existing_links.add(h["link"])
            added_count += 1
    
    # Tarihe göre tersten sırala (En yeni haber en üstte)
    try:
        existing_data["haberler"].sort(key=lambda x: datetime.strptime(x["tarih"], "%d.%m.%Y"), reverse=True)
    except:
        pass

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    print(f"İşlem bitti. {added_count} yeni haber arşive eklendi.")

if __name__ == "__main__":
    raw_news = get_recent_news()
    if raw_news:
        print(f"{len(raw_news)} haber analiz ediliyor...")
        ai_results = filter_with_ai(raw_news)
        save_and_merge(ai_results)
    else:
        print("Son 24 saatte yeni haber bulunamadı.")
