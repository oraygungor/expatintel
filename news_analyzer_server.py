import uvicorn
import os
import httpx
import json
import socket
import ipaddress
from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse, urljoin
from collections import defaultdict

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
from docx.shared import Pt, RGBColor

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
    raw_text: Optional[str] = None
    available_categories: Optional[List[str]] = None


class AnalyzeResponse(BaseModel):
    headline: str
    source: str
    summary: str
    expat_significance: str
    score: int
    category: str
    final_url: Optional[str] = None


class ExportItem(BaseModel):
    input_url: str
    analysis: AnalyzeResponse


# --- Word Helpers ---
def add_hyperlink(paragraph, text: str, url: str):
    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")

    color = OxmlElement('w:color')
    color.set(qn('w:val'), '0563C1')
    rPr.append(color)

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
    if not t: return ""
    if t[-1] in ".!?…": return t
    return t + "."

# --- Security & Fetching ---
def validate_url_security(url: str):
    try:
        parsed = urlparse(url)
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz URL formatı")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Sadece HTTP ve HTTPS protokolleri desteklenir.")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="Host bulunamadı.")
    return

async def fetch_url_content(url: str) -> tuple[str, str]:
    current_url = url
    async with httpx.AsyncClient(timeout=TIMEOUT, verify=True) as client:
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return resp.text, str(resp.url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"İçerik alınamadı: {str(e)}")

def extract_main_text(html_content: str) -> str:
    extracted = trafilatura.extract(html_content, include_comments=False, include_tables=False)
    return extracted[:15000] if extracted else ""

# --- Analyze Endpoint ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_news(request: AnalyzeRequest):
    
    clean_text = ""
    final_url = request.url

    if request.raw_text and len(request.raw_text.strip()) > 10:
        clean_text = request.raw_text[:15000]
        print("Manuel metin kullanılıyor.")
    else:
        print(f"Fetching: {request.url}")
        html_content, final_url = await fetch_url_content(request.url)
        clean_text = await run_in_threadpool(extract_main_text, html_content)

        if not clean_text or len(clean_text) < 50:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            clean_text = soup.get_text()[:5000]
    
    if len(clean_text) < 50:
        raise HTTPException(status_code=400, detail="Anlamlı metin bulunamadı.")

    # Kategori Listesi Hazırlığı
    categories_str = ""
    if request.available_categories and len(request.available_categories) > 0:
        cat_list = ", ".join([f"'{c}'" for c in request.available_categories])
        categories_str = f"Mevcut kategoriler: [{cat_list}]. Bu listeden en uygun olanı seç. Hiçbiri uymuyorsa 'Genel' seç."
    else:
        categories_str = "Kategori olarak 'Genel', 'Ekonomi', 'Politika', 'Vize', 'Sosyal Yaşam' gibi genel bir başlık seç."

    client = AsyncOpenAI(api_key=request.api_key)

    # Prompt Güncellemesi: Emojinin başlığın başında olduğundan emin oluyoruz.
    system_prompt = f"""
    Sen Danimarka'da yaşayan Türk expat'lar (Türkiye'den gelip Danimarka'da çalışan profesyoneller, göçmenler ve öğrenciler) için içerik üreten kıdemli bir haber analisti ve editörsün.
    
    GÖREV TANIMI VE KURALLAR:
    1. İÇERİK ANALİZİ: Metni oku ve özellikle Türk vatandaşlarının (ve genel olarak AB dışı vatandaşların) Danimarka'daki yaşam, çalışma, vize, oturum veya yasal durumlarına olan etkisini stratejik olarak değerlendir.
    2. BAŞLIK VE ÖZET AYRIMI (KRİTİK KURAL): 'headline' (başlık) ve 'summary' (özet) içerikleri birbirini tekrar etmemelidir. Başlık kısa ve dikkat çekici bir kanca olmalı; özet ise başlıkta kullanılan kelimeleri veya kalıpları kopyalamadan, haberin detaylarını bağımsız bir şekilde açıklamalıdır.
    3. TÜRK EXPAT ODAKLI PUANLAMA (ÖNEMLİ): Haberin Türkleri doğrudan veya dolaylı olarak ne kadar ilgilendirdiğini analiz et. Türk vatandaşlarını doğrudan etkileyen yasal değişiklikler, AB dışı (Non-EU) vize/oturum güncellemeleri veya Türkiye-Danimarka bağlamındaki haberlere çok daha yüksek puan (score) ver. Sadece AB vatandaşlarını ilgilendiren veya yerel/küçük çaplı Danimarka haberlerine düşük puan ver.
    4. KATEGORİZASYON: Haberi şu kategorilerden en uygun olanına ata: {categories_str}
    5. TON VE DİL: Profesyonel, nesnel, net ve kolay anlaşılır bir Türkçe kullan.
    6. ÇIKTI FORMATI: Sadece belirtilen JSON şemasına kesin olarak uyan yapılandırılmış veri üret.
    """

    news_schema = {
        "name": "news_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "headline": {
                    "type": "string", 
                    "description": "Dikkat çekici, vurucu ve kısa Türkçe başlık. Başlığın en başına konuyu yansıtan tek bir emoji koy. (Özetteki cümle yapısını ve kelimeleri tekrar etme)."
                },
                "source": {
                    "type": "string", 
                    "description": "Haberin alındığı kurumun veya platformun orijinal adı (Örn: DR, The Local Denmark, Ny i Danmark)."
                },
                "summary": {
                    "type": "string", 
                    "description": "Haberin ana hatlarını ve detaylarını aktaran 2-3 cümlelik profesyonel Türkçe özet. Başlıktaki ifadeleri tekrar etmeden haberin temel bağlamını ver."
                },
                "expat_significance": {
                    "type": "string", 
                    "description": "Bu haberin bir Türk expat'ın hayatını doğrudan nasıl etkileyeceğine dair 1-2 cümlelik nokta atışı analiz (Örn: 'Türk vatandaşları için çalışma izni süreçlerini hızlandırabilir', 'Sadece AB vatandaşlarını kapsıyor, Türkleri etkilemiyor')."
                },
                "score": {
                    "type": "integer", 
                    "description": "Haberin Türk expatlar için kritiklik seviyesi (1: Çok düşük ilgi/Sadece AB vatandaşlarını ilgilendiren haber, 10: Türkleri doğrudan etkileyen acil ve hayati değişiklik)."
                },
                "category": {
                    "type": "string", 
                    "description": "Sunulan listeden seçilmiş en isabetli kategori adı."
                }
            },
            "required": ["headline", "source", "summary", "expat_significance", "score", "category"],
            "additionalProperties": False,
        },
    }

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Hatası: {str(e)}")


# --- Export DOCX Endpoint (GÜNCELLENDİ) ---
@app.post("/export-docx")
async def export_docx(items: List[ExportItem]):
    if not items:
        raise HTTPException(status_code=400, detail="Boş liste.")

    doc = Document()
    
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

    # Sıralama: Kategoriye göre, sonra puana göre (yüksekten düşüğe)
    sorted_items = sorted(items, key=lambda x: (x.analysis.category or "Diğer", -x.analysis.score))

    from itertools import groupby
    
    for category, group_iter in groupby(sorted_items, key=lambda x: x.analysis.category or "Diğer"):
        # Kategori Başlığı - ARTIK BÜYÜK HARF DEĞİL (Upper kaldırıldı)
        h = doc.add_heading(category, level=1)
        run = h.runs[0]
        run.font.color.rgb = RGBColor(0, 0, 0)
        
        group_list = list(group_iter)
        
        for item in group_list:
            a = item.analysis
            
            # Başlık - PUAN KALDIRILDI, sadece başlık (emoji zaten AI tarafından başa ekleniyor)
            h2 = doc.add_heading(a.headline, level=3)
            h2.paragraph_format.space_before = Pt(12)
            
            # Özet
            summary = ensure_sentence_ends(a.summary)
            p = doc.add_paragraph()
            p.add_run(summary + " ")
            
            # Kaynak Linki
            p.add_run("(Kaynak: ")
            source_label = (a.source or "Link").strip()
            link = (a.final_url or item.input_url or "").strip()
            
            if link:
                add_hyperlink(p, source_label, link)
            else:
                p.add_run(source_label)
            p.add_run(")")
            
            # Expat Notu KISMI SİLİNDİ (Artık Word'de görünmeyecek)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="expatintel_grouped.docx"'}
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
