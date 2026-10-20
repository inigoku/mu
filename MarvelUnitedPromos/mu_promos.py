from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import re
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import imagehash
import requests
from bs4 import BeautifulSoup
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm
from docx.table import Table
from docx.text.paragraph import Paragraph

FIELDS = [
    "season", "expansion", "personaje", "role",
    "page_url", "original_source_url", "image_url", "final_url",
    "local_file", "figurita_url", "figurita_local_file",
    "status", "width", "height", "bytes",
    "sha256", "phash", "score", "alternatives_json",
    "notes", "updated_at"
]

HALL_PAGES = {
    "Temporada 1": "https://hallofunited.com/season-1-kickstarter-stretch-goals/",
    "Temporada 2": "https://hallofunited.com/season-2-kickstarter-stretch-goals/",
    "Temporada 3": "https://hallofunited.com/season-3-kickstarter-stretch-goals/",
}

CARD_DATABASE = "https://hallofunited.com/marvel-united-card-database/"

DENY_AUTOMATION = {
    "kickstarter.com", "www.kickstarter.com",
    "artstation.com", "www.artstation.com"
}

PLAYWRIGHT_ALLOWED = {
    "hallofunited.com", "bigchildcreatives.com",
    "www.cmon.com", "cmon.com", "www.ontabletop.com"
}

HOST_DELAY = {
    "hallofunited.com": 0.4,
    "bigchildcreatives.com": 3.0,
    "www.cmon.com": 3.0,
    "cmon.com": 3.0,
    "boardgamegeek.com": 5.0,
    "cf.geekdo-images.com": 5.0,
}

USER_AGENT = (
    "MarvelUnitedPersonalPromoCatalog/2.0 "
    "(personal-reference; replace-with-your-email)"
)

MIN_BYTES = 15_000
MIN_SIDE = 250
MAX_BYTES = 30 * 1024 * 1024
LOG = logging.getLogger("mu-promos")
LAST_REQUEST: dict[str, float] = {}
ROBOTS: dict[str, tuple[RobotFileParser | None, str]] = {}


def make_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,image/*;q=0.9,*/*;q=0.5"
    })
    return s


HTTP = make_session()


def host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def wait_for_host(url: str, extra: float = 0.0) -> None:
    h = host(url)
    delay = max(HOST_DELAY.get(h, 1.0), extra or 0)
    elapsed = time.monotonic() - LAST_REQUEST.get(h, 0)
    if elapsed < delay:
        time.sleep(delay - elapsed)
    LAST_REQUEST[h] = time.monotonic()


def robots_allowed(url: str):
    h = host(url)
    if h in DENY_AUTOMATION:
        return False, 0, "site_policy_denied"

    p = urlparse(url)
    root = f"{p.scheme}://{p.netloc}"

    if root not in ROBOTS:
        robots_url = root + "/robots.txt"
        try:
            wait_for_host(robots_url)
            r = HTTP.get(robots_url, timeout=20)
            if r.status_code == 404:
                ROBOTS[root] = (None, "missing")
            elif r.ok:
                rp = RobotFileParser()
                rp.parse(r.text.splitlines())
                ROBOTS[root] = (rp, "loaded")
            else:
                ROBOTS[root] = (None, f"http_{r.status_code}")
        except requests.RequestException as exc:
            ROBOTS[root] = (None, f"error:{exc}")

    rp, state = ROBOTS[root]
    if rp is None:
        return state == "missing", 0, state

    delay = rp.crawl_delay(USER_AGENT) or rp.crawl_delay("*") or 0
    return rp.can_fetch(USER_AGENT, url), delay, state


def get(url: str, stream: bool = False) -> requests.Response:
    allowed, delay, reason = robots_allowed(url)
    if not allowed:
        raise RuntimeError(f"Acceso automático omitido: {reason}")

    wait_for_host(url, delay)
    r = HTTP.get(url, stream=stream, timeout=(20, 75), allow_redirects=True)

    if r.status_code in (401, 403, 429):
        raise RuntimeError(f"Acceso detenido sin reintentar: HTTP {r.status_code}")

    r.raise_for_status()
    return r


def slug(value: str) -> str:
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE)
    return value.strip("._")[:110] or "sin_nombre"


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower().replace("’", "'").replace("'", "")).strip()


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def original_wordpress_url(url: str) -> str:
    """Quita parámetros ?w=700 y formatos de miniatura WordPress."""
    p = urlparse(url)
    if "hallofunited.com" in p.netloc:
        return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    return url


def download_bytes(url: str) -> tuple[bytes, str]:
    r = get(url, stream=True)
    content_type = r.headers.get("Content-Type", "").split(";")[0].lower()

    if not content_type.startswith("image/"):
        raise ValueError(f"Content-Type no válido: {content_type}")

    data = bytearray()
    for chunk in r.iter_content(64 * 1024):
        data.extend(chunk)
        if len(data) > MAX_BYTES:
            raise ValueError("Imagen demasiado grande")

    return bytes(data), r.url


def evaluate_plastic_miniature(data: bytes) -> dict:
    """Evalúa si una imagen es un render 3D de figurita (monocromática, sin bordes negros de carta)."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        w, h = im.size
        rgb = im.convert("RGB")
        pixels = list(rgb.getdata())
        total = max(1, w * h)
        black_pct = sum(1 for r, g, b in pixels if max(r, g, b) < 45) / total * 100.0

        hsv_pixels = list(rgb.convert("HSV").getdata())
        colored = [hue // 16 for hue, sat, val in hsv_pixels if sat > 50 and val > 50]
        if colored:
            top_hue_pct = max(Counter(colored).values()) / len(colored) * 100.0
        else:
            # Figurita gris (e.g. Grey Hulk)
            top_hue_pct = 90.0

        # Comprobar si ya tiene vista doble (delante + detrás) mirando la columna central vs laterales
        # Si hay figura en izquierda y derecha con hueco central, es vista doble.
        non_white_left = 0
        non_white_mid = 0
        non_white_right = 0
        for y in range(0, h, 4):
            row_off = y * w
            for x in range(int(w * 0.10), int(w * 0.38), 4):
                r, g, b = pixels[row_off + x]
                if min(r, g, b) < 215 and max(r, g, b) - min(r, g, b) > 15:
                    non_white_left += 1
            for x in range(int(w * 0.44), int(w * 0.56), 4):
                r, g, b = pixels[row_off + x]
                if min(r, g, b) < 215 and max(r, g, b) - min(r, g, b) > 15:
                    non_white_mid += 1
            for x in range(int(w * 0.62), int(w * 0.90), 4):
                r, g, b = pixels[row_off + x]
                if min(r, g, b) < 215 and max(r, g, b) - min(r, g, b) > 15:
                    non_white_right += 1

        is_Plastic = (black_pct < 1.0 and top_hue_pct > 72.0) or (black_pct < 0.35)
        score = top_hue_pct - 25.0 * black_pct
        if non_white_left > 25 and non_white_right > 25 and non_white_mid < min(non_white_left, non_white_right) * 0.45:
            score += 20.0
            is_dual = True
        else:
            is_dual = False

        return {
            "width": w,
            "height": h,
            "black_pct": round(black_pct, 3),
            "top_hue_pct": round(top_hue_pct, 1),
            "is_plastic": is_Plastic,
            "is_dual": is_dual,
            "score": round(score, 2),
        }


def save_jpeg(data: bytes, target: Path) -> tuple[int, int, str, str]:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        w, h = image.size
        # Si tiene transparencia (PNG/WEBP), poner fondo blanco
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            bg = Image.new("RGB", (w, h), (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            rgb = bg
        else:
            rgb = image.convert("RGB")

        out = io.BytesIO()
        rgb.save(out, "JPEG", quality=94, optimize=True)
        jpg_bytes = out.getvalue()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(jpg_bytes)
        digest = hashlib.sha256(jpg_bytes).hexdigest()
        perceptual = str(imagehash.phash(rgb))
        return w, h, digest, perceptual


def stitch_two_images_horizontally(data1: bytes, data2: bytes, target: Path) -> tuple[int, int]:
    with Image.open(io.BytesIO(data1)) as im1, Image.open(io.BytesIO(data2)) as im2:
        im1.load()
        im2.load()
        rgb1 = im1.convert("RGB")
        rgb2 = im2.convert("RGB")
        h = max(rgb1.height, rgb2.height)
        w1 = int(rgb1.width * (h / rgb1.height))
        w2 = int(rgb2.width * (h / rgb2.height))
        r1 = rgb1.resize((w1, h), Image.Resampling.LANCZOS)
        r2 = rgb2.resize((w2, h), Image.Resampling.LANCZOS)
        combined = Image.new("RGB", (w1 + w2, h), (255, 255, 255))
        combined.paste(r1, (0, 0))
        combined.paste(r2, (w1, 0))
        out = io.BytesIO()
        combined.save(out, "JPEG", quality=94, optimize=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out.getvalue())
        return w1 + w2, h


def add_hyperlink(paragraph, text: str, url: str) -> None:
    rid = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rid)

    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1769AA")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.extend([color, underline])
    run.append(props)

    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def build_word(manifest: Path, output: Path) -> None:
    rows = read_manifest(manifest)
    doc = Document()

    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = section.bottom_margin = Cm(1)
    section.left_margin = section.right_margin = Cm(1)

    doc.add_heading("Catálogo promocional Marvel United", 0)
    doc.add_paragraph(
        "Seasons 1, 2 y 3. Fichas de personaje y figuritas (vista delantera y trasera) "
        "almacenadas físicamente en el documento; enlaces conservados para atribución y trazabilidad."
    )

    current_group = None
    table = None
    cell_index = 3

    for row in rows:
        group = (row["season"], row["expansion"])

        if group != current_group:
            doc.add_heading(row["season"], level=1)
            doc.add_heading(row["expansion"], level=2)
            table = doc.add_table(rows=0, cols=3)
            current_group = group
            cell_index = 3

        if cell_index >= 3:
            cells = table.add_row().cells
            cell_index = 0

        cell = cells[cell_index]
        cell_index += 1

        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

        local_ficha = Path(row.get("local_file", ""))
        if local_ficha.is_file():
            p.add_run().add_picture(str(local_ficha), width=Cm(6.8))
        else:
            r = p.add_run("FICHA NO LOCALIZADA")
            r.bold = True

        local_mini = Path(row.get("figurita_local_file", ""))
        if local_mini.is_file():
            p_mini = cell.add_paragraph()
            p_mini.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_mini.add_run().add_picture(str(local_mini), width=Cm(6.8))

        name = cell.add_paragraph()
        name.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = name.add_run(row["personaje"])
        run.bold = True

        source = row.get("page_url") or row.get("original_source_url")
        if source:
            link = cell.add_paragraph()
            link.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_hyperlink(link, "Ficha en Hall of United", source)

    doc.save(output)
    verify_word(output, rows)


def verify_word(path: Path, rows: list[dict]) -> None:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        media = [x for x in names if x.startswith("word/media/")]
        document = z.read("word/document.xml").decode("utf-8")
        rels = z.read("word/_rels/document.xml.rels").decode("utf-8")

    embedded = document.count("<a:blip")
    external_images = re.findall(
        r'Type="[^"]+/image"[^>]+TargetMode="External"', rels)

    available_ficha = sum(
        bool(r.get("local_file")) and Path(r["local_file"]).is_file()
        for r in rows
    )
    available_mini = sum(
        bool(r.get("figurita_local_file")) and Path(r["figurita_local_file"]).is_file()
        for r in rows
    )

    if external_images:
        raise RuntimeError("Existen imágenes vinculadas externamente")
    if available_ficha and not media:
        raise RuntimeError("No se encontraron recursos en word/media")

    print(f"Word: {path}")
    print(f"Personajes totales: {len(rows)}")
    print(f"Con ficha incrustada: {available_ficha}")
    print(f"Con figurita incrustada: {available_mini}")
    print(f"Referencias embebidas totales: {embedded}")
    print(f"Archivos internos word/media: {len(media)}")
    print("Relaciones de imagen externas: 0")


def build_html(manifest: Path, output: Path) -> None:
    rows = read_manifest(manifest)
    items = []
    for idx, r in enumerate(rows):
        ficha_rel = (r.get("local_file") or "").replace("\\", "/")
        mini_rel = (r.get("figurita_local_file") or "").replace("\\", "/")
        items.append({
            "id": idx + 1,
            "season": r.get("season", ""),
            "expansion": r.get("expansion", ""),
            "personaje": r.get("personaje", ""),
            "page_url": r.get("page_url") or r.get("original_source_url") or "",
            "ficha": ficha_rel,
            "figurita": mini_rel,
            "ficha_url": r.get("image_url", ""),
            "figurita_url": r.get("figurita_url", ""),
        })

    data_json = json.dumps(items, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Catálogo Marvel United — Fichas y Figuritas</title>
  <style>
    :root {{
      --bg: #0f141c;
      --bg-panel: #18202c;
      --bg-card: #1d2636;
      --bg-card-hover: #253146;
      --border: #2e3b52;
      --text: #f0f4f8;
      --text-muted: #9fb0c7;
      --accent: #e62429;
      --accent-hover: #ff3b40;
      --badge-bg: #2a364b;
      --gold: #f5c518;
      --card-min: 320px;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(15, 20, 28, 0.94);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--border);
      padding: 0.9rem 1.5rem;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }}
    .header-inner {{
      max-width: 1600px;
      margin: 0 auto;
      display: flex;
      flex-wrap: wrap;
      gap: 0.9rem;
      align-items: center;
      justify-content: space-between;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    .brand-logo {{
      background: var(--accent);
      color: #fff;
      font-weight: 900;
      letter-spacing: 0.06em;
      padding: 0.3rem 0.65rem;
      border-radius: 4px;
      font-size: 1.05rem;
      text-transform: uppercase;
    }}
    .brand-title h1 {{
      font-size: 1.15rem;
      font-weight: 700;
    }}
    .brand-title span {{
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.65rem;
      align-items: center;
      flex: 1;
      justify-content: flex-end;
    }}
    .search-wrap {{
      position: relative;
      flex: 1 1 260px;
      max-width: 420px;
    }}
    .search-wrap input {{
      width: 100%;
      padding: 0.55rem 2.2rem 0.55rem 0.9rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg-panel);
      color: var(--text);
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.15s;
    }}
    .search-wrap input:focus {{
      border-color: var(--accent);
    }}
    .clear-btn {{
      position: absolute;
      right: 0.5rem;
      top: 50%;
      transform: translateY(-50%);
      background: none;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 1.1rem;
      display: none;
      padding: 0.2rem;
    }}
    .clear-btn:hover {{
      color: var(--text);
    }}
    select, .btn-toggle {{
      padding: 0.52rem 0.75rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg-panel);
      color: var(--text);
      font-size: 0.88rem;
      cursor: pointer;
      outline: none;
    }}
    select:focus, .btn-toggle:focus {{
      border-color: var(--accent);
    }}
    .stats-bar {{
      max-width: 1600px;
      margin: 0.55rem auto 0;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      font-size: 0.83rem;
      color: var(--text-muted);
      gap: 0.5rem;
    }}
    .season-pills {{
      display: flex;
      gap: 0.4rem;
      flex-wrap: wrap;
    }}
    .pill {{
      background: var(--badge-bg);
      color: var(--text-muted);
      border: 1px solid transparent;
      padding: 0.28rem 0.7rem;
      border-radius: 999px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.15s;
      user-select: none;
    }}
    .pill:hover {{
      color: var(--text);
      border-color: var(--border);
    }}
    .pill.active {{
      background: var(--accent);
      color: #fff;
      font-weight: 600;
    }}
    main {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 1.25rem 1.5rem 3rem;
    }}
    .group-section {{
      margin-bottom: 2.2rem;
    }}
    .group-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.45rem;
      margin-bottom: 1rem;
    }}
    .group-title {{
      display: flex;
      align-items: center;
      gap: 0.65rem;
      flex-wrap: wrap;
    }}
    .season-tag {{
      background: var(--accent);
      color: #fff;
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.18rem 0.55rem;
      border-radius: 4px;
      text-transform: uppercase;
    }}
    .expansion-title {{
      font-size: 1.2rem;
      font-weight: 700;
      color: var(--text);
    }}
    .group-count {{
      font-size: 0.82rem;
      color: var(--text-muted);
    }}
    .cards-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(var(--card-min), 1fr));
      gap: 1.1rem;
    }}
    .char-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    .char-card:hover {{
      transform: translateY(-2px);
      border-color: #485c7e;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }}
    .card-header {{
      padding: 0.7rem 0.9rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.5rem;
      border-bottom: 1px solid var(--border);
      background: rgba(0, 0, 0, 0.18);
    }}
    .char-name {{
      font-size: 1.02rem;
      font-weight: 700;
      color: #fff;
    }}
    .char-meta {{
      font-size: 0.74rem;
      color: var(--text-muted);
    }}
    .ext-link {{
      color: var(--gold);
      text-decoration: none;
      font-size: 0.78rem;
      font-weight: 600;
      white-space: nowrap;
      padding: 0.22rem 0.5rem;
      border-radius: 6px;
      background: rgba(245, 197, 24, 0.1);
      border: 1px solid rgba(245, 197, 24, 0.25);
    }}
    .ext-link:hover {{
      background: rgba(245, 197, 24, 0.22);
    }}
    .card-images {{
      padding: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
      flex: 1;
      justify-content: center;
    }}
    .img-box {{
      position: relative;
      background: #fff;
      border-radius: 8px;
      overflow: hidden;
      cursor: zoom-in;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .img-box img {{
      width: 100%;
      height: auto;
      display: block;
      object-fit: contain;
    }}
    .img-label {{
      position: absolute;
      bottom: 6px;
      right: 6px;
      background: rgba(15, 20, 28, 0.78);
      color: #fff;
      font-size: 0.68rem;
      padding: 2px 6px;
      border-radius: 4px;
      pointer-events: none;
    }}
    .empty-state {{
      text-align: center;
      padding: 4rem 1rem;
      color: var(--text-muted);
    }}
    .empty-state h3 {{
      font-size: 1.25rem;
      color: var(--text);
      margin-bottom: 0.5rem;
    }}
    /* Lightbox Modal */
    .modal-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(7, 10, 15, 0.92);
      z-index: 1000;
      display: none;
      flex-direction: column;
      justify-content: space-between;
      padding: 1rem;
      backdrop-filter: blur(6px);
    }}
    .modal-backdrop.open {{
      display: flex;
    }}
    .modal-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      max-width: 1400px;
      width: 100%;
      margin: 0 auto;
      padding: 0.4rem 0.5rem;
      gap: 1rem;
      flex-wrap: wrap;
    }}
    .modal-title {{
      display: flex;
      flex-direction: column;
    }}
    .modal-title h2 {{
      font-size: 1.25rem;
      color: #fff;
    }}
    .modal-title span {{
      font-size: 0.85rem;
      color: var(--text-muted);
    }}
    .modal-actions {{
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }}
    .modal-btn {{
      background: var(--bg-panel);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.45rem 0.85rem;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.85rem;
      text-decoration: none;
    }}
    .modal-btn:hover, .modal-btn.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .modal-body {{
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 1.25rem;
      overflow: auto;
      padding: 0.5rem;
      max-width: 1500px;
      width: 100%;
      margin: 0 auto;
    }}
    .modal-media {{
      display: flex;
      gap: 1.2rem;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      max-height: 82vh;
    }}
    .modal-media img {{
      max-height: 78vh;
      max-width: min(46vw, 720px);
      width: auto;
      height: auto;
      border-radius: 10px;
      background: #fff;
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
      object-fit: contain;
    }}
    .modal-media.single img {{
      max-width: min(85vw, 950px);
    }}
    .nav-arrow {{
      background: rgba(255, 255, 255, 0.1);
      color: #fff;
      border: 1px solid var(--border);
      width: 44px;
      height: 44px;
      border-radius: 50%;
      font-size: 1.3rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }}
    .nav-arrow:hover {{
      background: var(--accent);
    }}
    .modal-footer {{
      text-align: center;
      font-size: 0.78rem;
      color: var(--text-muted);
      padding-top: 0.25rem;
    }}
    @media (max-width: 768px) {{
      .modal-media img {{
        max-width: 85vw;
        max-height: 40vh;
      }}
      .modal-media.single img {{
        max-height: 72vh;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div class="brand">
        <div class="brand-logo">Marvel United</div>
        <div class="brand-title">
          <h1>Catálogo de Fichas y Figuritas</h1>
          <span>Hall of United — Temporadas 1, 2 y 3</span>
        </div>
      </div>

      <div class="controls">
        <div class="search-wrap">
          <input
            type="search"
            id="searchInput"
            placeholder="Buscar personaje, expansión o temporada... (Pulsa /)"
            autocomplete="off"
          />
          <button class="clear-btn" id="clearSearch" title="Limpiar búsqueda">&times;</button>
        </div>

        <select id="expansionSelect" title="Filtrar por expansión">
          <option value="">Todas las expansiones</option>
        </select>

        <select id="viewSelect" title="Imágenes a mostrar">
          <option value="both">Ficha + Figurita</option>
          <option value="ficha">Solo Fichas</option>
          <option value="figurita">Solo Figuritas</option>
        </select>

        <select id="sizeSelect" title="Tamaño de cuadrícula">
          <option value="240px">Vista Compacta</option>
          <option value="320px" selected>Vista Normal</option>
          <option value="420px">Vista Grande</option>
        </select>
      </div>
    </div>

    <div class="stats-bar">
      <div class="season-pills" id="seasonPills">
        <button class="pill active" data-season="">Todas</button>
        <button class="pill" data-season="Temporada 1">Temporada 1</button>
        <button class="pill" data-season="Temporada 2">Temporada 2</button>
        <button class="pill" data-season="Temporada 3">Temporada 3</button>
      </div>
      <div id="resultsCount">Mostrando 286 de 286 personajes</div>
    </div>
  </header>

  <main id="catalogContainer"></main>

  <div class="modal-backdrop" id="lightbox" role="dialog" aria-modal="true">
    <div class="modal-top">
      <div class="modal-title">
        <h2 id="lbName">Personaje</h2>
        <span id="lbMeta">Temporada · Expansión</span>
      </div>
      <div class="modal-actions">
        <button class="modal-btn active" data-lbview="both">Ficha + Figurita</button>
        <button class="modal-btn" data-lbview="ficha">Solo Ficha</button>
        <button class="modal-btn" data-lbview="figurita">Solo Figurita</button>
        <a id="lbSourceLink" class="modal-btn" href="#" target="_blank" rel="noopener">Hall of United ↗</a>
        <button class="modal-btn" id="lbClose" title="Cerrar (Esc)">✕ Cerrar</button>
      </div>
    </div>
    <div class="modal-body">
      <button class="nav-arrow" id="lbPrev" title="Anterior (←)">&#8592;</button>
      <div class="modal-media" id="lbMedia"></div>
      <button class="nav-arrow" id="lbNext" title="Siguiente (→)">&#8594;</button>
    </div>
    <div class="modal-footer">
      Usa las flechas <strong>←</strong> / <strong>→</strong> para cambiar de personaje, <strong>1/2/3</strong> para alternar vista y <strong>Esc</strong> para salir.
    </div>
  </div>

  <script>
    const DATA = {data_json};

    const searchInput = document.getElementById("searchInput");
    const clearSearch = document.getElementById("clearSearch");
    const expansionSelect = document.getElementById("expansionSelect");
    const viewSelect = document.getElementById("viewSelect");
    const sizeSelect = document.getElementById("sizeSelect");
    const seasonPills = document.getElementById("seasonPills");
    const resultsCount = document.getElementById("resultsCount");
    const catalogContainer = document.getElementById("catalogContainer");

    const lightbox = document.getElementById("lightbox");
    const lbName = document.getElementById("lbName");
    const lbMeta = document.getElementById("lbMeta");
    const lbMedia = document.getElementById("lbMedia");
    const lbSourceLink = document.getElementById("lbSourceLink");
    const lbClose = document.getElementById("lbClose");
    const lbPrev = document.getElementById("lbPrev");
    const lbNext = document.getElementById("lbNext");
    const lbViewBtns = document.querySelectorAll("[data-lbview]");

    let selectedSeason = "";
    let filteredList = [...DATA];
    let currentLbIndex = -1;
    let currentLbView = "both";

    function normalize(str) {{
      return (str || "")
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\\u0300-\\u036f]/g, "");
    }}

    function updateExpansionOptions() {{
      const currentVal = expansionSelect.value;
      const counts = new Map();
      for (const item of DATA) {{
        if (selectedSeason && item.season !== selectedSeason) continue;
        counts.set(item.expansion, (counts.get(item.expansion) || 0) + 1);
      }}
      expansionSelect.innerHTML = '<option value="">Todas las expansiones</option>';
      for (const [exp, count] of counts.entries()) {{
        const opt = document.createElement("option");
        opt.value = exp;
        opt.textContent = `${{exp}} (${{count}})`;
        expansionSelect.appendChild(opt);
      }}
      if (counts.has(currentVal)) {{
        expansionSelect.value = currentVal;
      }} else {{
        expansionSelect.value = "";
      }}
    }}

    function render() {{
      const q = normalize(searchInput.value.trim());
      clearSearch.style.display = q ? "block" : "none";
      const expFilter = expansionSelect.value;
      const viewMode = viewSelect.value;

      filteredList = DATA.filter((item) => {{
        if (selectedSeason && item.season !== selectedSeason) return false;
        if (expFilter && item.expansion !== expFilter) return false;
        if (q) {{
          const hay = normalize(`${{item.personaje}} ${{item.expansion}} ${{item.season}}`);
          if (!hay.includes(q)) return false;
        }}
        return true;
      }});

      resultsCount.textContent = `Mostrando ${{filteredList.length}} de ${{DATA.length}} personajes`;

      if (filteredList.length === 0) {{
        catalogContainer.innerHTML = `
          <div class="empty-state">
            <h3>No se encontraron personajes</h3>
            <p>Prueba con otro término de búsqueda o cambia los filtros seleccionados.</p>
          </div>
        `;
        return;
      }}

      // Group by (season, expansion)
      const groups = [];
      let lastKey = null;
      for (const item of filteredList) {{
        const key = `${{item.season}}|||${{item.expansion}}`;
        if (key !== lastKey) {{
          groups.push({{
            season: item.season,
            expansion: item.expansion,
            items: [],
          }});
          lastKey = key;
        }}
        groups[groups.length - 1].items.push(item);
      }}

      const frag = document.createDocumentFragment();
      let flatIndex = 0;

      for (const g of groups) {{
        const sec = document.createElement("section");
        sec.className = "group-section";

        const header = document.createElement("div");
        header.className = "group-header";
        header.innerHTML = `
          <div class="group-title">
            <span class="season-tag">${{g.season}}</span>
            <h2 class="expansion-title">${{g.expansion}}</h2>
          </div>
          <span class="group-count">${{g.items.length}} personaje${{g.items.length === 1 ? "" : "s"}}</span>
        `;
        sec.appendChild(header);

        const grid = document.createElement("div");
        grid.className = "cards-grid";

        for (const item of g.items) {{
          const itemIdx = flatIndex++;
          const card = document.createElement("article");
          card.className = "char-card";

          const imagesHtml = [];
          if (viewMode === "both" || viewMode === "ficha") {{
            imagesHtml.push(`
              <div class="img-box" data-idx="${{itemIdx}}" data-prefer="both">
                <img src="${{item.ficha}}" alt="Ficha de ${{item.personaje}}" loading="lazy" />
                <span class="img-label">Ficha</span>
              </div>
            `);
          }}
          if (viewMode === "both" || viewMode === "figurita") {{
            imagesHtml.push(`
              <div class="img-box" data-idx="${{itemIdx}}" data-prefer="both">
                <img src="${{item.figurita}}" alt="Figurita de ${{item.personaje}}" loading="lazy" />
                <span class="img-label">Figurita</span>
              </div>
            `);
          }}

          card.innerHTML = `
            <div class="card-header">
              <div>
                <div class="char-name">${{item.personaje}}</div>
                <div class="char-meta">${{item.season}} · ${{item.expansion}}</div>
              </div>
              <a class="ext-link" href="${{item.page_url}}" target="_blank" rel="noopener" title="Abrir en Hall of United">Hall of United ↗</a>
            </div>
            <div class="card-images">
              ${{imagesHtml.join("")}}
            </div>
          `;
          grid.appendChild(card);
        }}

        sec.appendChild(grid);
        frag.appendChild(sec);
      }}

      catalogContainer.innerHTML = "";
      catalogContainer.appendChild(frag);
    }}

    function openLightbox(index) {{
      if (index < 0 || index >= filteredList.length) return;
      currentLbIndex = index;
      const item = filteredList[currentLbIndex];
      lbName.textContent = item.personaje;
      lbMeta.textContent = `${{item.season}} · ${{item.expansion}} (${{currentLbIndex + 1}} de ${{filteredList.length}})`;
      lbSourceLink.href = item.page_url || "#";

      lbMedia.className = "modal-media" + (currentLbView === "both" ? "" : " single");
      const imgs = [];
      if (currentLbView === "both" || currentLbView === "ficha") {{
        imgs.push(`<img src="${{item.ficha}}" alt="Ficha de ${{item.personaje}}" />`);
      }}
      if (currentLbView === "both" || currentLbView === "figurita") {{
        imgs.push(`<img src="${{item.figurita}}" alt="Figurita de ${{item.personaje}}" />`);
      }}
      lbMedia.innerHTML = imgs.join("");
      lightbox.classList.add("open");
      document.body.style.overflow = "hidden";
    }}

    function closeLightbox() {{
      lightbox.classList.remove("open");
      document.body.style.overflow = "";
    }}

    function setLbView(mode) {{
      currentLbView = mode;
      lbViewBtns.forEach((btn) => {{
        btn.classList.toggle("active", btn.dataset.lbview === mode);
      }});
      if (currentLbIndex >= 0) openLightbox(currentLbIndex);
    }}

    catalogContainer.addEventListener("click", (e) => {{
      const box = e.target.closest(".img-box");
      if (!box) return;
      const idx = parseInt(box.dataset.idx, 10);
      openLightbox(idx);
    }});

    lbClose.addEventListener("click", closeLightbox);
    lightbox.addEventListener("click", (e) => {{
      if (e.target === lightbox || e.target === lbMedia) closeLightbox();
    }});
    lbPrev.addEventListener("click", () => {{
      if (filteredList.length === 0) return;
      openLightbox((currentLbIndex - 1 + filteredList.length) % filteredList.length);
    }});
    lbNext.addEventListener("click", () => {{
      if (filteredList.length === 0) return;
      openLightbox((currentLbIndex + 1) % filteredList.length);
    }});

    lbViewBtns.forEach((btn) => {{
      btn.addEventListener("click", () => setLbView(btn.dataset.lbview));
    }});

    searchInput.addEventListener("input", render);
    clearSearch.addEventListener("click", () => {{
      searchInput.value = "";
      searchInput.focus();
      render();
    }});
    expansionSelect.addEventListener("change", render);
    viewSelect.addEventListener("change", render);
    sizeSelect.addEventListener("change", () => {{
      document.documentElement.style.setProperty("--card-min", sizeSelect.value);
    }});

    seasonPills.addEventListener("click", (e) => {{
      const pill = e.target.closest(".pill");
      if (!pill) return;
      selectedSeason = pill.dataset.season;
      document.querySelectorAll(".pill").forEach((p) => p.classList.toggle("active", p === pill));
      updateExpansionOptions();
      render();
    }});

    document.addEventListener("keydown", (e) => {{
      if (lightbox.classList.contains("open")) {{
        if (e.key === "Escape") closeLightbox();
        else if (e.key === "ArrowLeft") lbPrev.click();
        else if (e.key === "ArrowRight") lbNext.click();
        else if (e.key === "1") setLbView("both");
        else if (e.key === "2") setLbView("ficha");
        else if (e.key === "3") setLbView("figurita");
        return;
      }}
      if (e.key === "/" && document.activeElement !== searchInput) {{
        e.preventDefault();
        searchInput.focus();
      }}
    }});

    updateExpansionOptions();
    render();
  </script>
</body>
</html>
"""
    output.write_text(html_content, encoding="utf-8")
    print(f"HTML generado: {output} ({len(items)} personajes)")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("mu_promos.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("word")
    p.add_argument("--manifest", type=Path, default=Path("manifest_promos.csv"))
    p.add_argument("--output", type=Path, default=Path("Marvel_United_Promos_Offline.docx"))

    p_html = commands.add_parser("html")
    p_html.add_argument("--manifest", type=Path, default=Path("manifest_promos.csv"))
    p_html.add_argument("--output", type=Path, default=Path("index.html"))

    args = parser.parse_args()

    if args.command == "word":
        build_word(args.manifest, args.output)
    elif args.command == "html":
        build_html(args.manifest, args.output)


if __name__ == "__main__":
    main()