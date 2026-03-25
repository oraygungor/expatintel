import json
import feedparser
import time
import os
import re
from datetime import datetime, timedelta
from openai import OpenAI
from googlenewsdecoder import gnewsdecoder

# --- GÜVENLİK ---
API_KEY = os.getenv("OPENAI_API_KEY")

RSS_FEEDS = [
    "https://feeds.thelocal.com/rss/dk",
    "https://news.google.com/rss/search?q=Denmark+when:1d&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Denmark+when:1d&hl=da&gl=DK&ceid=DK:da",
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
    time_limit = now - timedelta(hours=24)
    
    print(f"\n--- HABER TARAMA BAŞLADI ({now.strftime('%H:%M:%S')}) ---")
    
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
                    "summary": entry.get("summary", "")[:600] # Analiz için yeterli uzunlukta özet alıyoruz
                })
        
        print(f"[{source_name[:35]:<35}] Taranan: {feed_total:<3} | Yeni: {new_on_feed}")
    
    return articles

def filter_with_ai(news_list):
    if not news_list: return {"haberler": []}
    if not API_KEY:
        print("HATA: OpenAI API Anahtarı bulunamadı!")
        return {"haberler": []}
        
    client = OpenAI(api_key=API_KEY)
    
    news_dict = {item['link']: item for item in news_list}
    context = ""
    for idx, item in enumerate(news_list):
        context += f"Kaynak: {item['source']}\nBaşlık: {item['title']}\nTarih: {item['date']}\nLink: {item['link']}\nİçerik/Özet: {item['summary']}\n\n"

    # --- 1. AŞAMA: HABERLERİ FİLTRELEME VE PUANLAMA ---
    prompt_1 = f"""
    Aşağıdaki haberlerden Danimarka'daki expat'ları, özellikle Türk expatları ilgilendirenleri seç. 
    
    ÖNEMLİ KURALLAR:
    1. Sonuçları TÜRKÇE olarak dön.
    2. Sadece başlık, kaynak, link, tarih ve expat_puani (1-10 arası) alanlarını doldur.
    3. Puanı sadece 8 ve üzeri olanları seç (Yasal düzenlemeler, göçmenlik, vergiler, Danimarka-Türkiye ilişkileri vb. baz alınarak).
    4. 'tarih' alanına KESİNLİKLE sadece GG.AA.YYYY formatında tarih yaz (Örn: {datetime.now().strftime("%d.%m.%Y")}).
    
    JSON FORMATI:
    {{
      "haberler": [
        {{
          "baslik": "Türkçe Başlık",
          "kaynak": "Kaynak",
          "link": "Link",
          "tarih": "{datetime.now().strftime("%d.%m.%Y")}",
          "expat_puani": 9
        }}
      ]
    }}
    
    HABER HAVUZU:
    {context}
    """

    try:
        print("\nAI Aşama 1: Haberler filtreleniyor...")
        response_1 = client.chat.completions.create(
            model="gpt-5.4", 
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "Sen profesyonel bir haber seçici ve analistisin. Sadece JSON dönersin."},
                {"role": "user", "content": prompt_1}
            ]
        )
        selected_data = json.loads(response_1.choices[0].message.content)
        selected_news = selected_data.get("haberler", [])
    except Exception as e:
        print(f"AI Aşama 1 Hatası: {e}")
        return {"haberler": []}

    # --- 2. AŞAMA: HER BİR HABER İÇİN ÖZET VE ETKİ ANALİZİ ---
    final_news = []
    
    valid_news = []
    for h in selected_news:
        try:
            if int(h.get("expat_puani", 0)) >= 8:
                valid_news.append(h)
        except:
            pass

    if not valid_news:
        print("AI Aşama 1: 8 puan ve üzeri expat haberi bulunamadı. 2. aşama atlanıyor.")
        return {"haberler": []}

    print(f"AI Aşama 2: Kriterleri geçen {len(valid_news)} haber için özet ve etki analizi yapılıyor...")
    
    for haber in valid_news:
        link = haber.get("link", "")
        orijinal_haber = news_dict.get(link, {})
        orijinal_ozet = orijinal_haber.get("summary", "")
        orijinal_baslik = orijinal_haber.get("title", "")
        
        prompt_2 = f"""
        Aşağıdaki Danimarka haberi için iki farklı metin oluştur:
        1. Haberin tamamen tarafsız, sade bir dille yazılmış, net ve anlaşılır bir özeti (En fazla 2 cümle).
        2. Bu gelişmenin Danimarka'da yaşayan Türk expatlara (beyaz yakalılar, öğrenciler, çalışanlar vb.) yönelik yakın ve ileri vadeli olası etkilerinin analizi (En fazla 3 cümle, profesyonel, akıcı ve öngörüsel bir dille).
        
        Türkçe Başlık: {haber.get('baslik')}
        Orijinal Başlık: {orijinal_baslik}
        Haberin Ham Özeti/İçeriği: {orijinal_ozet}
        
        Lütfen SADECE aşağıdaki JSON formatında yanıt ver:
        {{
          "ozet": "Haberin tarafsız ve sade özeti buraya yazılacak...",
          "ai_analiz": "Türk expatlara potansiyel etkisi buraya yazılacak..."
        }}
        """
        
        try:
            response_2 = client.chat.completions.create(
                model="gpt-5.4",
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": "Sen stratejik bir haber özetleyici ve sosyo-ekonomik etki analistisin. Sadece JSON dönersin."},
                    {"role": "user", "content": prompt_2}
                ]
            )
            analiz_sonucu = json.loads(response_2.choices[0].message.content)
            
            haber["ozet"] = analiz_sonucu.get("ozet", "Özet oluşturulamadı.")
            haber["ai_analiz"] = analiz_sonucu.get("ai_analiz", "Etki analizi oluşturulamadı.")
            final_news.append(haber)
            
            print(f" - Özet ve Analiz tamamlandı: {haber.get('baslik')[:30]}...")
        except Exception as e:
            print(f"AI Aşama 2 Hatası ({link}): {e}")
            haber["ozet"] = "Özet oluşturulurken bir hata oluştu."
            haber["ai_analiz"] = "Bu haberin etki analizi oluşturulurken bir hata oluştu."
            final_news.append(haber)

    return {"haberler": final_news}

def safe_date_parse(date_str):
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

    two_weeks_ago = datetime.now() - timedelta(days=14)
    filtered_haberler = [
        h for h in data.get("haberler", [])
        if safe_date_parse(h.get("tarih", "")) >= two_weeks_ago
    ]
    
    silinen_sayisi = len(data.get("haberler", [])) - len(filtered_haberler)
    data["haberler"] = filtered_haberler

    existing_links = {h["link"] for h in data.get("haberler", [])}
    added_count = 0
    
    for h in new_data.get("haberler", []):
        if h["link"] not in existing_links:
            if "tarih" not in h or not h["tarih"] or len(h["tarih"]) < 8:
                h["tarih"] = datetime.now().strftime("%d.%m.%Y")
            data["haberler"].append(h)
            added_count += 1
    
    data["haberler"].sort(key=lambda x: safe_date_parse(x.get("tarih", "")), reverse=True)

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n--- İŞLEM BAŞARIYLA TAMAMLANDI ---")
    if silinen_sayisi > 0:
        print(f"Temizlenen Eski Haber Sayısı (>14 gün): {silinen_sayisi}")
    print(f"Yeni Eklenen Haber: {added_count} | Toplam Arşiv: {len(data['haberler'])}")

if __name__ == "__main__":
    existing_links = set()
    if os.path.exists("daily_news.json"):
        try:
            with open("daily_news.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_links = {h["link"] for h in data.get("haberler", [])}
        except:
            pass

    raw_news = get_recent_news()
    
    new_raw_news = [n for n in raw_news if n["link"] not in existing_links]
    
    if new_raw_news:
        print(f"\nVeritabanında olmayan {len(new_raw_news)} yeni haber AI'ya gönderiliyor...")
        ai_output = filter_with_ai(new_raw_news)
        save_and_merge(ai_output)
    else:
        print("\nSon 24 saatte eklenecek yeni içerik bulunamadı.")
