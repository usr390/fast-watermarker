from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import io
import os
from zipfile import ZipFile, ZIP_DEFLATED
import tempfile
import subprocess
import shutil

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "Ubuntu-Regular.ttf")

IMAGE_MIME_PREFIXES = ("image/",)
VIDEO_MIME_PREFIXES = ("video/",)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm"}

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
async def upload_multi(
    files: List[UploadFile] = File(...), 
    text: Optional[str] = Form(None), 
    logo: Optional[UploadFile] = File(None), 
    logo_scale: float = Form(0.35), 
    logo_opacity: float = Form(0.9),
    position: str = Form("br")):
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images allowed.")
    
    has_logo = bool(logo and logo.filename)
    if not has_logo and not text:
        return StreamingResponse(
            io.BytesIO(b'{"detail":"Provide a logo or text"}'),
            media_type="application/json",
            headers={"Content-Disposition": "inline"},
        )

    logo_bytes = None
    
    if logo is not None and logo.filename:
        logo_bytes = await logo.read()

    zip_buf = io.BytesIO()
    with ZipFile(zip_buf, "w", ZIP_DEFLATED) as zf:
        for f in files:
            raw = await f.read()
            fname = f.filename or "file"
            try:
                if _is_image(f):
                    if has_logo:
                        out = add_image_watermark(raw, logo_bytes, scale=logo_scale, opacity=logo_opacity)
                    else:
                        out = add_text_watermark(raw, text)
                    out_name = f"watermarked_{safe_name(fname, suffix='.jpg')}"
                    zf.writestr(out_name, out.getvalue())

                elif _is_video(f):
                    if has_logo:
                        out = watermark_video_with_logo(raw, logo_bytes, scale=logo_scale, opacity=logo_opacity, position=position)
                    else:
                        out = watermark_video_with_text(raw, text, position=position)
                    out_name = f"watermarked_{safe_name(fname, suffix='.mp4')}"
                    zf.writestr(out_name, out.getvalue())

                else:
                    zf.writestr(f"SKIPPED_{safe_name(fname, suffix='.txt')}", b"Unsupported file type.")
            except Exception as ex:
                zf.writestr(f"ERROR_{safe_name(fname, suffix='.txt')}", f"Failed to process: {ex}".encode("utf-8"))

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

def _is_image(f: UploadFile) -> bool:
    ct = (f.content_type or "").lower()
    if ct.startswith(IMAGE_MIME_PREFIXES): return True
    _, ext = os.path.splitext((f.filename or "").lower())
    return ext in IMAGE_EXTS

def _is_video(f: UploadFile) -> bool:
    ct = (f.content_type or "").lower()
    if ct.startswith(VIDEO_MIME_PREFIXES): return True
    _, ext = os.path.splitext((f.filename or "").lower())
    return ext in VIDEO_EXTS

def _which_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH. Install it (brew/apt) or run in Docker.")
    
def _pos_expr(position: str, margin: int = 16) -> str:
    pos = (position or "br").lower()
    if pos == "tl": return f"x={margin}:y={margin}"
    if pos == "tr": return f"x=W-w-{margin}:y={margin}"
    if pos == "bl": return f"x={margin}:y=H-h-{margin}"
    if pos == "center": return "x=(W-w)/2:y=(H-h)/2"
    return f"x=W-w-{margin}:y=H-h-{margin}"

def watermark_video_with_logo(
    video_bytes: bytes,
    logo_bytes: bytes,
    scale: float = 0.25,
    opacity: float = 0.85,
    position: str = "center",
) -> io.BytesIO:
    _which_ffmpeg()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as lf, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as of:
        vf.write(video_bytes); vf.flush()
        lf.write(logo_bytes);  lf.flush()
        in_video, in_logo, out_path = vf.name, lf.name, of.name

    video_w = _probe_video_width(in_video)
    target_w = max(1, int(video_w * float(scale)))

    filter_complex = (
        f"[1:v]scale={target_w}:-1:flags=lanczos,format=rgba,colorchannelmixer=aa={opacity}[lg];"
        f"[0:v][lg]overlay=x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2:eval=init[outv]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", in_video, "-i", in_logo,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        out_path,
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True)
    try:
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")
        with open(out_path, "rb") as f:
            buf = io.BytesIO(f.read()); buf.seek(0); return buf
    finally:
        for p in (in_video, in_logo, out_path):
            try: os.remove(p)
            except: pass

def watermark_video_with_text(
    video_bytes: bytes,
    text: str,
    position: str = "br",
    font_size: int = 28,
    font_color: str = "white",
    shadow_color: str = "black",
    shadow_x: int = 2,
    shadow_y: int = 2,
    margin: int = 16,
) -> io.BytesIO:
    _which_ffmpeg()
    if not os.path.exists(FONT_PATH):
        raise RuntimeError(f"Font not found at {FONT_PATH}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as of:
        vf.write(video_bytes); vf.flush()
        in_video, out_path = vf.name, of.name

    pos = (position or "br").lower()
    if pos == "tl": dx, dy = f"{margin}", f"{margin}"
    elif pos == "tr": dx, dy = f"(w-tw)-{margin}", f"{margin}"
    elif pos == "bl": dx, dy = f"{margin}", f"(h-th)-{margin}"
    elif pos == "center": dx, dy = "(w-tw)/2", "(h-th)/2"
    else: dx, dy = f"(w-tw)-{margin}", f"(h-th)-{margin}"

    draw = (
        f"drawtext=fontfile='{FONT_PATH}':text='{text}':"
        f"fontsize={font_size}:fontcolor={font_color}:"
        f"shadowcolor={shadow_color}:shadowx={shadow_x}:shadowy={shadow_y}:"
        f"x={dx}:y={dy}"
    )

    cmd = [
        "ffmpeg","-y",
        "-i", in_video,
        "-vf", draw,
        "-map","0:v:0","-map","0:a?",
        "-c:v","libx264","-preset","veryfast","-crf","23",
        "-c:a","aac","-b:a","128k",
        "-movflags","+faststart",
        "-pix_fmt","yuv420p",
        out_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(out_path, "rb") as f:
            buf = io.BytesIO(f.read())
            buf.seek(0)
            return buf
    finally:
        for p in (in_video, out_path):
            try: os.remove(p)
            except: pass
def _overlay_pos_expr(position: str, margin: int = 16) -> str:
    pos = (position or "br").lower()
    if pos == "tl":
        return f"x={margin}:y={margin}"
    if pos == "tr":
        return f"x=main_w-overlay_w-{margin}:y={margin}"
    if pos == "bl":
        return f"x={margin}:y=main_h-overlay_h-{margin}"
    if pos == "center":
        return "x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2"
    # default br
    return f"x=main_w-overlay_w-{margin}:y=main_h-overlay_h-{margin}"

def _probe_video_width(path: str) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width",
        "-of", "csv=p=0", path,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0 or not proc.stdout.strip().isdigit():
        raise RuntimeError(f"ffprobe failed to get width: {proc.stderr or proc.stdout}")
    return int(proc.stdout.strip())