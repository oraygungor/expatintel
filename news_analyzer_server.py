import uvicorn
import os
import httpx
import json
import socket
import ipaddress
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from urllib.parse import urlparse
from openai import OpenAI
import trafilatura

# --- Configuration ---
app = FastAPI()

# CORS: Production'da 'allow_origins' listesine sadece kendi frontend domainini eklemelisin.
# allow_origins=["*"] ile allow_credentials=True ÇALIŞMAZ. Credentials (Cookie) kullanılmıyorsa False yapılır.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production için: ["https://senin-github-io-adresin.github.io"]
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB Limit
TIMEOUT = 10.0  # Saniye

# --- Data Models ---
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

# --- SSRF Protection Logic ---
def validate_url_security(url: str):
    """
    URL'nin güvenli olup olmadığını (SSRF) kontrol eder.
    Private IP'lere, Loopback'e ve geçersiz protokollere erişimi engeller.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz URL formatı")

    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Sadece HTTP ve HTTPS protokolleri desteklenir.")

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Host bulunamadı.")

    # DNS Çözümleme ve IP Kontrolü
    try:
        # socket.gethostbyname bloklayıcıdır, ancak basitlik ve güvenlik için burada kullanıyoruz.
        # Async ortamda çok yoğun yükte burası threadpool'a alınabilir.
        ip_str = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_str)
    except Exception:
        raise HTTPException(status_code=400, detail="DNS çözümlenemedi veya geçersiz Host.")

    # Yasaklı IP Aralıkları
    if (ip.is_private or 
        ip.is_loopback or 
        ip.is_link_local or 
        ip.is_multicast or 
        ip.is_reserved):
        raise HTTPException(status_code=403, detail="Erişim engellendi: Hedef IP güvenli değil (SSRF Koruması).")

# --- Async Content Fetching ---
async def fetch_url_content(url: str) -> str:
    """
    httpx ile asenkron olarak URL'i çeker.
    Redirect takibi yapar ama SSRF için her redirect adımını kontrol etmek gerekir.
    Basitlik adına httpx'in güvenli ayarlarına güveniyoruz ama production'da redirect hook kullanılmalı.
    """
    # Önce güvenlik kontrolü
    validate_url_security(url)

    async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT, verify=False) as client:
        try:
            # Stream response to enforce size limit
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                
                # Check Content-Type (Sadece metin tabanlı içerikler)
                ct = response.headers.get("content-type", "").lower()
                if "text" not in ct and "html" not in ct and "json" not in ct:
                    raise HTTPException(status_code=400, detail=f"Desteklenmeyen içerik tipi: {ct}")

                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_CONTENT_LENGTH:
                        raise HTTPException(status_code=400, detail=f"İçerik boyutu sınırı ({MAX_CONTENT_LENGTH/1024/1024}MB) aşıldı.")
                
                return data.decode('utf-8', errors='ignore')

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Kaynak hatası: {e.response.status_code}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Bağlantı hatası: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"İçerik alınamadı: {str(e)}")

# --- Content Extraction (CPU Bound) ---
def extract_main_text(html_content: str) -> str:
    """
    Trafilatura kullanarak HTML'den temiz metin çıkarır.
    Bu işlem CPU yoğun olduğu için threadpool'da çalıştırılmalıdır.
    """
    extracted = trafilatura.extract(html_content, include_comments=False, include_tables=False)
    if not extracted:
        return ""
    return extracted[:15000] # Token limiti için kırp

# --- Endpoint ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_news(request: AnalyzeRequest):
    # 1. URL Çekme (IO Bound -> Async)
    print(f"Fetching: {request.url}")
    html_content = await fetch_url_content(request.url)
    
    if not html_content:
        raise HTTPException(status_code=400, detail="Sayfa içeriği boş.")

    # 2. İçerik Temizleme (CPU Bound -> run_in_threadpool)
    clean_text = await run_in_threadpool(extract_main_text, html_content)
    
    if not clean_text or len(clean_text) < 50:
        # Trafilatura başarısız olursa ham HTML'den basit text'e dön (fallback)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        clean_text = soup.get_text()[:5000]
        if len(clean_text) < 50:
             raise HTTPException(status_code=400, detail="Anlamlı metin çıkarılamadı.")

    # 3. OpenAI Analizi
    client = OpenAI(api_key=request.api_key)
    
    system_prompt = """
    Sen Danimarka'daki expat'lar için uzman bir haber asistanısın.
    Sana bir haberin İÇERİĞİ verilecek.
    
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

    # Model Fallback ve İstek Yönetimi
    try:
        # Önce Structured Output (Schema) dene
        try:
            completion = client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"URL: {request.url}\n\nMETİN:\n{clean_text}"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": news_schema
                }
            )
            content = completion.choices[0].message.content
            return json.loads(content)
            
        except Exception as schema_error:
            print(f"Schema failed, retrying with JSON mode: {schema_error}")
            # Desteklenmiyorsa JSON Mode dene
            completion = client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": system_prompt + " Yanıtı sadece JSON objesi olarak ver."},
                    {"role": "user", "content": f"URL: {request.url}\n\nMETİN:\n{clean_text}"}
                ],
                response_format={"type": "json_object"}
            )
            content = completion.choices[0].message.content
            # Basit bir validasyon yapıp döndür
            data = json.loads(content)
            # Eğer eksik alan varsa default ata (basit kurtarma)
            return AnalyzeResponse(
                headline=data.get("headline", "Başlık Bulunamadı"),
                source=data.get("source", "Bilinmiyor"),
                summary=data.get("summary", "Özet çıkarılamadı."),
                expat_significance=data.get("expat_significance", "Bilgi yok."),
                score=data.get("score", 1)
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI İşlem Hatası: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Sunucu başlatılıyor... Port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
