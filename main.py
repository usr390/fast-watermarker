from typing import Optional, List
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import io
import os
from zipfile import ZipFile, ZIP_DEFLATED

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "Ubuntu-Regular.ttf")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(file: UploadFile = File(...), text: Optional[str] = Form(None), logo: Optional[UploadFile] = File(None), logo_scale: float = Form(0.2)):
    
    base_bytes = await file.read()
    if logo is not None and logo.filename:
        logo_bytes = await logo.read()
        output = add_image_watermark(base_bytes, logo_bytes, scale=logo_scale)
    else:
        if not text:
            return StreamingResponse(io.BytesIO(base_bytes), media_type="image/jpeg")
        output = add_text_watermark(base_bytes, text)

    return StreamingResponse(output, media_type="image/jpeg")

@app.post("/upload-multi")
async def upload_multi(files: List[UploadFile] = File(...), text: Optional[str] = Form(None), logo: Optional[UploadFile] = File(None), logo_scale: float = Form(0.35), logo_opacity: float = Form(0.9)):
    if (logo is None or not logo.filename) and not text:
        return StreamingResponse(
            io.BytesIO(b'{"detail":"Provide a logo or text"}'),
            media_type="application/json",
            headers={"Content-Disposition": 'inline'}
        )

    logo_bytes = None
    if logo is not None and logo.filename:
        logo_bytes = await logo.read()

    zip_buf = io.BytesIO()
    with ZipFile(zip_buf, "w", ZIP_DEFLATED) as zf:
        for f in files:
            base_bytes = await f.read()
            if logo_bytes:
                out = add_image_watermark(base_bytes, logo_bytes, scale=logo_scale, opacity=logo_opacity)
                out_name = f"watermarked_{safe_name(f.filename, suffix='.jpg')}"
            else:
                out = add_text_watermark(base_bytes, text)
                out_name = f"watermarked_{safe_name(f.filename, suffix='.jpg')}"

            zf.writestr(out_name, out.getvalue())

    zip_buf.seek(0)
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="watermarked.zip"'}
    )

def add_text_watermark(img_bytes: bytes, text: str) -> io.BytesIO:
    base = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGBA")
    W, H = base.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font_size = max(16, W // 20)
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = (W - tw) // 2
    y = (H - th) // 2

    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0, 128))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 180))

    watermarked = Image.alpha_composite(base, overlay).convert("RGB")

    buf = io.BytesIO()
    watermarked.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return buf

def add_image_watermark(img_bytes: bytes, logo_bytes: bytes, scale=0.2, opacity=0.85) -> io.BytesIO:
    base = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGBA")
    logo = ImageOps.exif_transpose(Image.open(io.BytesIO(logo_bytes))).convert("RGBA")


    W, H = base.size

    target_w = max(1, int(W * float(scale)))
    target_h = max(1, int(logo.height * (target_w / logo.width)))
    logo = logo.resize((target_w, target_h), Image.LANCZOS)

    if has_transparency(logo):
        logo = scale_existing_alpha(logo, opacity)
    else:
        logo = set_uniform_opacity(logo, opacity)

    W, H = base.size
    lw, lh = logo.size

    x = (W - lw) // 2
    y = (H - lh) // 2

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    overlay.paste(logo, (x, y), mask=logo)

    out_img = Image.alpha_composite(base, overlay).convert("RGB")
    buf = io.BytesIO()
    out_img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return buf

def has_transparency(im: Image.Image) -> bool:
    if im.mode in ("RGBA", "LA"):
        return im.getchannel("A").getextrema()[0] < 255
    return "transparency" in im.info

def scale_existing_alpha(im: Image.Image, opacity: float) -> Image.Image:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    r, g, b, a = im.split()
    a = ImageEnhance.Brightness(a).enhance(opacity)
    return Image.merge("RGBA", (r, g, b, a))

def set_uniform_opacity(im: Image.Image, opacity: float) -> Image.Image:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    r, g, b, _ = im.split()
    a = Image.new("L", im.size, int(255 * opacity))
    return Image.merge("RGBA", (r, g, b, a))


def safe_name(name: str, suffix: str = ".jpg") -> str:
    base, _sep, _ext = name.partition(".")
    base = base.strip().replace("/", "_").replace("\\", "_")
    base = base if base else "image"
    return base + suffix