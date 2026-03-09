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
    "https://news.google.com/rss/search?q=Denmark+(expats+OR+foreigners+OR+immigration)&hl=en-US&gl=US&ceid=US:en",
    "https://cphpost.dk/feed/",
    "https://www.thelocal.dk/feed"
]

def resolve_google_news_url(url: str) -> str:
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
    
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            pub_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            
            if not pub_date or pub_date >= time_limit:
                title = entry.get("title", "").strip()
                # Tarihi DD.MM.YYYY formatına çeviriyoruz
                date_str = pub_date.strftime("%d.%m.%Y") if pub_date else now.strftime("%d.%m.%Y")
                
                if title.lower() in seen_titles: continue
                
                final_link = resolve_google_news_url(entry.get("link", ""))
                if final_link in seen_links: continue

                seen_links.add(final_link)
                seen_titles.add(title.lower())
                
                articles.append({
                    "title": title,
                    "link": final_link,
                    "date": date_str,
                    "source": feed.feed.get("title", "News Source")
                })
    return articles

def filter_with_ai(news_list):
    if not news_list: return '{"haberler": []}'
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    context = ""
    for idx, item in enumerate(news_list):
        context += f"ID: {idx}\nBaşlık: {item['title']}\nTarih: {item['date']}\nLink: {item['link']}\n\n"

    prompt = f"""
    Aşağıdaki haberlerden Danimarka'daki expatları ilgilendirenleri seç. 
    Aynı haberi tekrar etme. Puanı 8 ve üzeri olanları JSON olarak dön.
    Haber tarihini 'tarih' anahtarıyla ekle.
    
    Model: gpt-5.4
    
    Haberler:
    {context}
    """

    response = client.chat.completions.create(
        model="gpt-5.4", # Kullanıcı isteği üzerine güncellendi
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "Sen sadece JSON çıktı veren bir haber editörüsün."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    news = get_recent_news()
    result_json = filter_with_ai(news)
    print(result_json)
    
    # İstersen sonucu bir dosyaya da kaydedebilirsin
    with open("daily_news.json", "w", encoding="utf-8") as f:
        f.write(result_json)
