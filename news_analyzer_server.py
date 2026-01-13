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
from urllib.parse import urlparse, urljoin
from openai import AsyncOpenAI  # Async client kullanımı
import trafilatura

# --- Configuration ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB Limit
TIMEOUT = 10.0
MAX_REDIRECTS = 5  # Maksimum yönlendirme sayısı

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
    final_url: str  # Yönlendirmeler sonrası gidilen son URL

# --- SSRF & DNS Protection Logic ---
def validate_url_security(url: str):
    """
    URL'nin güvenli olup olmadığını (SSRF) kontrol eder.
    DNS çözümlemesi yaparak Private, Loopback ve Reserved IP'leri engeller.
    IPv4 ve IPv6 destekler.
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

    # 1. Hostname bir IP literal mi? (Örn: 127.0.0.1 veya [::1])
    try:
        ip = ipaddress.ip_address(hostname)
        if (ip.is_private or ip.is_loopback or ip.is_link_local or 
            ip.is_multicast or ip.is_reserved):
            raise HTTPException(status_code=403, detail="Erişim engellendi: Hedef IP güvenli değil (SSRF).")
        return  # IP literal ve güvenli (public)
    except ValueError:
        pass  # IP değil, DNS çözümlemesine geç

    # 2. DNS Çözümleme (Tüm kayıtları kontrol et: A ve AAAA)
    try:
        # socket.getaddrinfo hem IPv4 hem IPv6 döndürür
        addr_info = socket.getaddrinfo(hostname, None)
        
        for res in addr_info:
            # res[4][0] IP adresini verir
            ip_str = res[4][0]
            ip = ipaddress.ip_address(ip_str)
            
            if (ip.is_private or ip.is_loopback or ip.is_link_local or 
                ip.is_multicast or ip.is_reserved):
                 raise HTTPException(status_code=403, detail=f"Erişim engellendi: Yasaklı IP çözümlendi ({ip_str}).")
                 
    except socket.gaierror:
        raise HTTPException(status_code=400, detail="DNS çözümlenemedi.")
    except Exception as e:
        # Production'da loglanmalı
        raise HTTPException(status_code=400, detail=f"Güvenlik kontrolü hatası: {str(e)}")

# --- Async Content Fetching (Secure) ---
async def fetch_url_content(url: str) -> tuple[str, str]:
    """
    httpx ile asenkron olarak URL'i çeker.
    Redirect'leri MANUEL takip eder ve her adımda validate_url_security çalıştırır.
    verify=True (SSL) zorunludur.
    """
    current_url = url
    
    # Asenkron Client (verify=True güvenlik için kritik)
    async with httpx.AsyncClient(timeout=TIMEOUT, verify=True) as client:
        for _ in range(MAX_REDIRECTS + 1):
            # Her adımda (redirect dahil) güvenlik kontrolü
            validate_url_security(current_url)

            try:
                # stream=True ile başlıkları alıp içeriği sonra indiriyoruz (Boyut kontrolü için)
                # follow_redirects=False -> Kontrolü biz yapıyoruz
                async with client.stream("GET", current_url, follow_redirects=False) as response:
                    
                    # Redirect Kontrolü (3xx)
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location")
                        if not location:
                            raise HTTPException(status_code=400, detail="Yönlendirme adresi bulunamadı.")
                        
                        # Yeni URL'i hesapla ve döngüye devam et
                        next_url = urljoin(current_url, location)
                        current_url = next_url
                        continue 

                    response.raise_for_status()
                    
                    # Content-Type Kontrolü (Genişletilmiş)
                    ct = response.headers.get("content-type", "").lower()
                    allowed_types = ["text/", "html", "xml", "json", "application/xhtml+xml", "application/xml"]
                    
                    if not any(t in ct for t in allowed_types):
                         raise HTTPException(status_code=400, detail=f"Desteklenmeyen içerik tipi: {ct}")

                    # Body İndirme ve Boyut Limiti
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_CONTENT_LENGTH:
                            raise HTTPException(status_code=400, detail=f"İçerik boyutu sınırı ({MAX_CONTENT_LENGTH/1024/1024}MB) aşıldı.")
                    
                    # Başarılı dönüş: (İçerik, Son URL)
                    return data.decode('utf-8', errors='ignore'), str(response.url)

            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=f"Kaynak hatası: {e.response.status_code}")
            except httpx.RequestError as e:
                raise HTTPException(status_code=400, detail=f"Bağlantı hatası: {str(e)}")
            except Exception as e:
                 if isinstance(e, HTTPException): raise e
                 raise HTTPException(status_code=400, detail=f"İçerik alınamadı: {str(e)}")

        raise HTTPException(status_code=400, detail="Maksimum yönlendirme sayısına ulaşıldı.")

# --- Content Extraction (CPU Bound) ---
def extract_main_text(html_content: str) -> str:
    # Trafilatura thread içinde çalıştırılacak
    extracted = trafilatura.extract(html_content, include_comments=False, include_tables=False)
    if not extracted:
        return ""
    return extracted[:15000]

# --- Endpoint ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_news(request: AnalyzeRequest):
    # 1. URL Çekme (IO Bound -> Async & SSRF Protected)
    print(f"Fetching: {request.url}")
    html_content, final_url = await fetch_url_content(request.url)
    
    if not html_content:
        raise HTTPException(status_code=400, detail="Sayfa içeriği boş.")

    # 2. İçerik Temizleme (CPU Bound -> run_in_threadpool)
    clean_text = await run_in_threadpool(extract_main_text, html_content)
    
    # Fallback: BeautifulSoup
    if not clean_text or len(clean_text) < 50:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        clean_text = soup.get_text()[:5000]
        if len(clean_text) < 50:
             raise HTTPException(status_code=400, detail="Anlamlı metin çıkarılamadı.")

    # 3. OpenAI Analizi (Async Client)
    client = AsyncOpenAI(api_key=request.api_key)
    
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

    try:
        # Structured Output (Schema)
        try:
            completion = await client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"URL: {final_url}\n\nMETİN:\n{clean_text}"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": news_schema
                }
            )
            content = completion.choices[0].message.content
            data = json.loads(content)
            
            # Final URL'i response'a ekle (Schema'da olmayabilir, manuel ekliyoruz)
            data["final_url"] = final_url
            return data
            
        except Exception as schema_error:
            print(f"Schema failed, retrying with JSON mode: {schema_error}")
            
            # Fallback: JSON Mode
            completion = await client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": system_prompt + " Yanıtı sadece JSON objesi olarak ver."},
                    {"role": "user", "content": f"URL: {final_url}\n\nMETİN:\n{clean_text}"}
                ],
                response_format={"type": "json_object"}
            )
            content = completion.choices[0].message.content
            data = json.loads(content)
            
            # Güvenli Tip Dönüşümü ve Clamp
            try:
                score = int(data.get("score", 1))
                score = max(1, min(10, score))
            except (ValueError, TypeError):
                score = 1

            return AnalyzeResponse(
                headline=data.get("headline", "Başlık Bulunamadı"),
                source=data.get("source", "Bilinmiyor"),
                summary=data.get("summary", "Özet çıkarılamadı."),
                expat_significance=data.get("expat_significance", "Bilgi yok."),
                score=score,
                final_url=final_url
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI İşlem Hatası: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Sunucu başlatılıyor... Port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
