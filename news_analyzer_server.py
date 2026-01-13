import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import os

# --- Yapılandırma ---
app = FastAPI()

# HTML dosyanızın bu sunucuyla konuşabilmesi için CORS izni veriyoruz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Güvenlik notu: Prodüksiyonda buraya sadece kendi domaininizi yazın
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Veri Modelleri ---
class AnalyzeRequest(BaseModel):
    url: str
    api_key: str
    model: str = "gpt-4o-mini"

class AnalyzeResponse(BaseModel):
    headline: str
    source: str
    summary: str
    expat_significance: str
    score: int

# --- Yardımcı Fonksiyon: Web Scraping ---
def fetch_and_clean_url(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Gereksiz etiketleri temizle (Scriptler, stiller, navigasyon vs.)
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
            
        # Metni al ve temizle
        text = soup.get_text()
        
        # Boşlukları düzenle
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Token limitini aşmamak için metni kırp (Örn: ilk 15.000 karakter)
        return text[:15000]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"URL okunamadı: {str(e)}")

# --- Endpoint ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_news(request: AnalyzeRequest):
    # 1. Adım: URL İçeriğini Çek
    print(f"Fetching: {request.url}")
    page_content = fetch_and_clean_url(request.url)
    
    if not page_content or len(page_content) < 50:
        raise HTTPException(status_code=400, detail="Sayfa içeriği boş veya okunamadı.")

    # 2. Adım: OpenAI ile Analiz Et
    client = OpenAI(api_key=request.api_key)
    
    system_prompt = """
    Sen Danimarka'daki expat'lar için uzman bir haber asistanısın.
    Sana bir haberin HAM METNİ verilecek.
    
    GÖREVLER:
    1. Metni oku ve anla.
    2. Expat'lar için önemini değerlendir.
    3. JSON formatında Türkçe çıktı ver.
    """
    
    news_schema = {
        "name": "news_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "headline": { "type": "string", "description": "İlgi çekici Türkçe başlık." },
                "source": { "type": "string", "description": "Haber kaynağının adı." },
                "summary": { "type": "string", "description": "Türkçe özet (2-3 cümle)." },
                "expat_significance": { "type": "string", "description": "Expatlar için neden önemli?" },
                "score": { "type": "integer", "description": "Önem puanı (1-10)", "minimum": 1, "maximum": 10 }
            },
            "required": ["headline", "source", "summary", "expat_significance", "score"],
            "additionalProperties": False
        }
    }

    try:
        completion = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {request.url}\n\nİÇERİK:\n{page_content}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": news_schema
            }
        )
        
        content = completion.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Hatası: {str(e)}")

if __name__ == "__main__":
    # Bulut ortamı için dinamik port ayarı
    port = int(os.environ.get("PORT", 8000))
    print(f"Sunucu başlatılıyor... Port: {port}")
    # Host 0.0.0.0 olmalı ki dış dünyadan erişilebilsin
    uvicorn.run(app, host="0.0.0.0", port=port)
