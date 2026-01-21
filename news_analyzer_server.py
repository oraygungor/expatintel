import uvicorn
import os
import httpx
import json
import socket
import ipaddress
from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse, urljoin

import trafilatura
from openai import AsyncOpenAI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE


# --- Configuration ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
TIMEOUT = 10.0
MAX_REDIRECTS = 5


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
    final_url: Optional[str] = None  # Redirect sonrası gidilen son URL


class ExportItem(BaseModel):
    input_url: str
    analysis: AnalyzeResponse


# --- Word Hyperlink Helper (GÜNCELLENDİ) ---
def add_hyperlink(paragraph, text: str, url: str):
    """
    Paragraph içine tıklanabilir hyperlink ekler.
    Word stil dosyasına güvenmek yerine, XML seviyesinde manuel olarak
    MAVİ RENK ve ALTI ÇİZGİ ekler, böylece link olduğu kesin belli olur.
    """
    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")

    # 1. Renk: Standart Link Mavisi (0563C1)
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '0563C1')
    rPr.append(color)

    # 2. Altı Çizili (Single Underline)
    u = OxmlElement('w:u')
    u.set(qn('w:val'), 'single')
    rPr.append(u)

    new_run.append(rPr)
    
    t = OxmlElement("w:t")
    t.text = text
    new_run.append(t)

    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink


def ensure_sentence_ends(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if t[-1] in ".!?…":
        return t
    return t + "."


# --- SSRF & DNS Protection Logic ---
def validate_url_security(url: str):
    try:
        parsed = urlparse(url)
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz URL formatı")

    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Sadece HTTP ve HTTPS protokolleri desteklenir.")

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Host bulunamadı.")

    try:
        ip = ipaddress.ip_address(hostname)
        if (ip.is_private or ip.is_loopback or ip.is_link_local or
            ip.is_multicast or ip.is_reserved):
            raise HTTPException(status_code=403, detail="Erişim engellendi: Hedef IP güvenli değil (SSRF).")
        return
    except ValueError:
        pass

    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for res in addr_info:
            ip_str = res[4][0]
            ip = ipaddress.ip_address(ip_str)
            if (ip.is_private or ip.is_loopback or ip.is_link_local or
                ip.is_multicast or ip.is_reserved):
                raise HTTPException(status_code=403, detail=f"Erişim engellendi: Yasaklı IP çözümlendi ({ip_str}).")
    except socket.gaierror:
        # DNS çözümlenemedi ama bazen local development veya özel setup'larda olabilir.
        # Yine de production güvenliği için hata fırlatıyoruz.
        raise HTTPException(status_code=400, detail="DNS çözümlenemedi.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Güvenlik kontrolü hatası: {str(e)}")


# --- Async Content Fetching (Secure) ---
async def fetch_url_content(url: str) -> tuple[str, str]:
    current_url = url

    async with httpx.AsyncClient(timeout=TIMEOUT, verify=True) as client:
        for _ in range(MAX_REDIRECTS + 1):
            validate_url_security(current_url)

            try:
                async with client.stream("GET", current_url, follow_redirects=False) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location")
                        if not location:
                            raise HTTPException(status_code=400, detail="Yönlendirme adresi bulunamadı.")
                        current_url = urljoin(current_url, location)
                        continue

                    response.raise_for_status()

                    ct = response.headers.get("content-type", "").lower()
                    allowed_types = ["text/", "html", "xml", "json", "application/xhtml+xml", "application/xml"]
                    if not any(t in ct for t in allowed_types):
                        raise HTTPException(status_code=400, detail=f"Desteklenmeyen içerik tipi: {ct}")

                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_CONTENT_LENGTH:
                            raise HTTPException(
                                status_code=400,
                                detail=f"İçerik boyutu sınırı ({MAX_CONTENT_LENGTH/1024/1024}MB) aşıldı."
                            )

                    return data.decode("utf-8", errors="ignore"), str(response.url)

            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=f"Kaynak hatası: {e.response.status_code}")
            except httpx.RequestError as e:
                raise HTTPException(status_code=400, detail=f"Bağlantı hatası: {str(e)}")
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise e
                raise HTTPException(status_code=400, detail=f"İçerik alınamadı: {str(e)}")

        raise HTTPException(status_code=400, detail="Maksimum yönlendirme sayısına ulaşıldı.")


# --- Content Extraction (CPU Bound) ---
def extract_main_text(html_content: str) -> str:
    extracted = trafilatura.extract(html_content, include_comments=False, include_tables=False)
    if not extracted:
        return ""
    return extracted[:15000]


# --- Analyze Endpoint ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_news(request: AnalyzeRequest):
    print(f"Fetching: {request.url}")

    html_content, final_url = await fetch_url_content(request.url)
    if not html_content:
        raise HTTPException(status_code=400, detail="Sayfa içeriği boş.")

    clean_text = await run_in_threadpool(extract_main_text, html_content)

    if not clean_text or len(clean_text) < 50:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        clean_text = soup.get_text()[:5000]
        if len(clean_text) < 50:
            raise HTTPException(status_code=400, detail="Anlamlı metin çıkarılamadı.")

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
                "headline": {"type": "string", "description": "İlgi çekici Türkçe başlık."},
                "source": {"type": "string", "description": "Haber kaynağının adı."},
                "summary": {"type": "string", "description": "Türkçe özet (2-3 cümle)."},
                "expat_significance": {"type": "string", "description": "Expatlar için neden önemli?"},
                "score": {"type": "integer", "description": "Önem puanı (1-10)", "minimum": 1, "maximum": 10},
            },
            "required": ["headline", "source", "summary", "expat_significance", "score"],
            "additionalProperties": False,
        },
    }

    try:
        try:
            completion = await client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"URL: {final_url}\n\nMETİN:\n{clean_text}"},
                ],
                response_format={"type": "json_schema", "json_schema": news_schema},
            )

            content = completion.choices[0].message.content
            data = json.loads(content)
            data["final_url"] = final_url
            return data

        except Exception as schema_error:
            print(f"Schema failed, retrying with JSON mode: {schema_error}")
            completion = await client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": system_prompt + " Yanıtı sadece JSON objesi olarak ver."},
                    {"role": "user", "content": f"URL: {final_url}\n\nMETİN:\n{clean_text}"},
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            data = json.loads(content)

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
                final_url=final_url,
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI İşlem Hatası: {str(e)}")


# --- Export DOCX Endpoint ---
@app.post("/export-docx")
async def export_docx(items: List[ExportItem]):
    """
    Frontend'den gelen globalResults listesi -> Word (.docx) indir.
    """
    if not items:
        raise HTTPException(status_code=400, detail="Boş liste gönderildi.")

    doc = Document()

    for item in items:
        a = item.analysis

        # Header 3: Title
        title = (a.headline or "Başlık Bulunamadı").strip()
        doc.add_heading(title, level=3)

        # Summary + (Kaynak: DR) aynı paragraf içinde
        summary = ensure_sentence_ends(a.summary or "")
        p = doc.add_paragraph()
        if summary:
            p.add_run(summary + " ")
        else:
            p.add_run("")

        p.add_run("(Kaynak: ")
        
        source_label = (a.source or "Kaynak").strip()
        
        # Link'i belirle: Backend'den gelen final_url varsa onu, yoksa user'ın girdiği input_url'i kullan.
        link = (a.final_url or item.input_url or "").strip()

        if not link:
            # Link yoksa fallback: sadece yaz
            p.add_run(source_label)
        else:
            # Varsa Mavi ve Altı Çizili Hyperlink ekle
            add_hyperlink(p, source_label, link)
            
        p.add_run(")")

        # Boş satır
        doc.add_paragraph("")

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="expatintel.docx"'}
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Sunucu başlatılıyor... Port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
