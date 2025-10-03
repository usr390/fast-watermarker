from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageEnhance, ImageOps
import io
import os
from zipfile import ZipFile, ZIP_DEFLATED
import tempfile
import subprocess
import shutil
import json
import base64
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv package is not installed. Run 'pip install python-dotenv'.")

dotenv_loaded = load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "Ubuntu-Regular.ttf")

IMAGE_MIME_PREFIXES = ("image/",)
VIDEO_MIME_PREFIXES = ("video/",)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm"}

MAX_REQUEST_BYTES = 600 * 1024 * 1024 # 600MB
MAX_IMAGE_MB = 25
MAX_IMAGE_PIXELS = 13_000_000              # 13MP
MAX_VIDEO_MB = 300
MAX_VIDEO_DURATION_SEC = 10 * 60


@app.middleware("http")
async def reject_large_requests(request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_REQUEST_BYTES:
        return HTMLResponse("Request too large", status_code=413)
    return await call_next(request)

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    # Public marketing/SEO page
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    # The actual tool UI
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/login")
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/pricing")
async def pricing(request: Request):
    return templates.TemplateResponse("pricing.html", {"request": request})

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    text: Optional[str] = Form(None),
    logo: Optional[UploadFile] = File(None),
    scale: float = Form(0.35),
    position: str = Form("center"),
    opacity: float = Form(0.9),
):
    """
    Single-file upload that matches /upload-multi behavior:
    - Accepts image OR video
    - Enforces same size/duration limits
    - Requires either logo or text (like /upload-multi)
    - Uses the same watermarking helpers as multi
    """
    # Normalize/validate inputs
    op = max(0.05, min(1.0, float(opacity)))
    has_logo = bool(logo and logo.filename)
    if not has_logo and not text:
        # Match /upload-multi's "provide a logo or text" behavior
        return StreamingResponse(
            io.BytesIO(b'{"detail":"Provide a logo or text"}'),
            media_type="application/json",
            headers={"Content-Disposition": "inline"},
        )

    raw = await file.read()

    # If a logo is provided, load it once
    logo_bytes = None
    if has_logo:
        logo_bytes = await logo.read()

    # Branch on file type (use the same helpers you use in /upload-multi)
    if _is_image(file):
        # Image validations (same as multi)
        with Image.open(io.BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im)
            if im.width * im.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    413,
                    f"{file.filename}: image too large (>{MAX_IMAGE_PIXELS} pixels).",
                )

        # Process
        if has_logo:
            out = add_image_watermark_ffmpeg(
                raw, logo_bytes, scale=scale, opacity=op, position=position
            )
        else:
            out = add_text_watermark_ffmpeg(
                raw, text, scale, position, op
            )

        # Name & respond (jpeg like multi)
        out_name = f"watermarked_{safe_name(file.filename or 'image', suffix='.jpg')}"
        headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
        return StreamingResponse(out, media_type="image/jpeg", headers=headers)

    elif _is_video(file):
        # Video validations (same as multi)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tf:
            tf.write(raw)
            tf.flush()
            size = os.path.getsize(tf.name)
            _reject_if_too_large(file.filename or "video", size, is_video=True)
            meta = _probe_video_meta(tf.name)
            if meta["duration"] > MAX_VIDEO_DURATION_SEC:
                raise HTTPException(
                    413,
                    f"{file.filename}: video longer than {MAX_VIDEO_DURATION_SEC//60} minutes.",
                )

        # Process
        if has_logo:
            out = watermark_video_with_logo(
                raw, logo_bytes, scale=scale, opacity=op, position=position
            )
        else:
            out = watermark_video_with_text(
                raw, text, scale=scale, position=position, opacity=op
            )

        # Name & respond (mp4 like multi)
        out_name = f"watermarked_{safe_name(file.filename or 'video', suffix='.mp4')}"
        headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
        return StreamingResponse(out, media_type="video/mp4", headers=headers)

    else:
        # Unsupported
        raise HTTPException(status_code=415, detail="Unsupported file type.")

@app.post("/upload-multi")
async def upload_multi(
    files: List[UploadFile] = File(...), 
    text: Optional[str] = Form(None), 
    logo: Optional[UploadFile] = File(None), 
    scale: float = Form(0.35), 
    opacity: float = Form(0.9),
    position: str = Form("center")):
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images allowed.")
    
    op = max(0.05, min(1.0, float(opacity)))
    
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

            if _is_image(f):
                with Image.open(io.BytesIO(raw)) as im:
                    im = ImageOps.exif_transpose(im)
                    if im.width * im.height > MAX_IMAGE_PIXELS:
                        raise HTTPException(413, f"{f.filename}: image too large (>{MAX_IMAGE_PIXELS} pixels).")
            elif _is_video(f):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tf:
                    tf.write(raw); tf.flush()
                    sz = os.path.getsize(tf.name)
                    _reject_if_too_large(f.filename, sz, True)
                    meta = _probe_video_meta(tf.name)
                    if meta["duration"] > MAX_VIDEO_DURATION_SEC:
                        raise HTTPException(413, f"{f.filename}: video longer than {MAX_VIDEO_DURATION_SEC//60} minutes.")
            
            fname = f.filename or "file"
            try:
                if _is_image(f):
                    if has_logo:
                        out = add_image_watermark_ffmpeg(raw, logo_bytes, scale=scale, opacity=op, position=position)
                    else:
                        out = add_text_watermark_ffmpeg(raw, text, scale, position, op)
                    out_name = f"watermarked_{safe_name(fname, suffix='.jpg')}"
                    zf.writestr(out_name, out.getvalue())

                elif _is_video(f):
                    if has_logo:
                        out = watermark_video_with_logo(raw, logo_bytes, scale=scale, opacity=op, position=position)
                    else:
                        out = watermark_video_with_text(raw, text, scale=scale, position=position, opacity=op)
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

def _text_px_from_scale(base_w: int, scale: float) -> int:
    return max(18, int(base_w * 0.06 * (scale / 0.35)))


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

    pos_expr = _overlay_pos_expr(position)
    filter_complex = (
        f"[1:v]scale={target_w}:-1:flags=lanczos,format=rgba,colorchannelmixer=aa={opacity}[lg];"
        f"[0:v][lg]overlay={pos_expr}:eval=init[outv]"
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
    scale: float = 0.35,
    position: str = "center",
    font_color: str = "white",
    shadow_color: str = "black",
    shadow_x: int = 2,
    shadow_y: int = 2,
    margin: int = 16,
    opacity: float = 0.9
) -> io.BytesIO:
    _which_ffmpeg()
    if not os.path.exists(FONT_PATH):
        raise RuntimeError(f"Font not found at {FONT_PATH}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as of:
        vf.write(video_bytes); vf.flush()
        in_video, out_path = vf.name, of.name

    vid_w = _probe_video_width(in_video)
    font_size = max(18, int(vid_w * 0.06 * (scale / 0.35)))

    pos = (position or "center").lower()
    if pos == "tl": dx, dy = f"{margin}", f"{margin}"
    elif pos == "tr": dx, dy = f"(w-tw)-{margin}", f"{margin}"
    elif pos == "bl": dx, dy = f"{margin}", f"(h-th)-{margin}"
    elif pos == "center": dx, dy = "(w-tw)/2", "(h-th)/2"
    else: dx, dy = f"(w-tw)-{margin}", f"(h-th)-{margin}"

    op = max(0.05, min(1.0, float(opacity)))
    op_shadow = min(1.0, op * 0.7)

    safe_text = _escape_drawtext(text or "")

    draw = (
        f"drawtext=fontfile='{FONT_PATH}':text='{safe_text}':"
        f"fontsize={font_size}:fontcolor={font_color}@{op}:"
        f"shadowcolor={shadow_color}@{op_shadow}:shadowx={shadow_x}:shadowy={shadow_y}:"
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
    pos = (position or "center").lower()
    if pos == "tl":
        return f"x={margin}:y={margin}"
    if pos == "tr":
        return f"x=main_w-overlay_w-{margin}:y={margin}"
    if pos == "bl":
        return f"x={margin}:y=main_h-overlay_h-{margin}"
    if pos == "center":
        return "x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2"

    return f"x=main_w-overlay_w-{margin}:y=main_h-overlay_h-{margin}"

def _probe_video_width(path: str) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width",
        "-of", "json", path,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed to get width: {proc.stderr or proc.stdout}")
    try:
        data = json.loads(proc.stdout)
        width = int(data["streams"][0]["width"])
        return width
    except Exception:
        raise RuntimeError(f"ffprobe parse error: {proc.stdout}")

def _escape_drawtext(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'").replace('"', r'\"')



def _reject_if_too_large(filename: str, size_bytes: int, is_video: bool):
    mb = size_bytes / (1024*1024)
    if is_video and mb > MAX_VIDEO_MB:
        raise HTTPException(413, f"{filename}: exceeds {MAX_VIDEO_MB} MB")
    if not is_video and mb > MAX_IMAGE_MB:
        raise HTTPException(413, f"{filename}: exceeds {MAX_IMAGE_MB} MB")

def _probe_video_meta(path: str) -> dict:
    cmd = ["ffprobe","-v","error","-select_streams","v:0",
           "-show_entries","stream=width,height,duration",
           "-of","default=noprint_wrappers=1:nokey=0", path]
    p = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr or p.stdout}")
    meta = {}
    for line in p.stdout.splitlines():
        if "=" in line:
            k,v = line.split("=",1)
            meta[k.strip()] = v.strip()
    meta["width"] = int(float(meta.get("width","0") or 0))
    meta["height"] = int(float(meta.get("height","0") or 0))
    meta["duration"] = float(meta.get("duration","0") or 0.0)
    return meta

def add_image_watermark_ffmpeg(
    img_bytes: bytes, 
    logo_bytes: bytes, 
    scale: float = 0.35, 
    opacity: float = 0.85, 
    position: str = "center"
) -> io.BytesIO:
    _which_ffmpeg()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_f, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as logo_f, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as out_f:
        img_f.write(img_bytes); img_f.flush()
        logo_f.write(logo_bytes); logo_f.flush()
        in_img, in_logo, out_path = img_f.name, logo_f.name, out_f.name

    img_w = _probe_image_width(in_img)
    target_w = max(1, int(img_w * float(scale)))
    pos_expr = _overlay_pos_expr(position)
    filter_complex = (
        f"[1:v]scale={target_w}:-1:flags=fast_bilinear,format=rgba,colorchannelmixer=aa={opacity}[lg];"
        f"[0:v][lg]overlay={pos_expr}[outv]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", in_img, "-i", in_logo,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-q:v", "2",  # High quality JPEG
        out_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(out_path, "rb") as f:
            buf = io.BytesIO(f.read())
            buf.seek(0)
            return buf
    finally:
        for p in (in_img, in_logo, out_path):
            try: os.remove(p)
            except: pass

def _probe_image_width(path: str) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width",
        "-of", "json", path,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed to get image width: {proc.stderr or proc.stdout}")
    try:
        data = json.loads(proc.stdout)
        width = int(data["streams"][0]["width"])
        return width
    except Exception:
        raise RuntimeError(f"ffprobe parse error: {proc.stdout}")
    
def _escape_drawtext(s: str) -> str:
    if s is None:
        return ""
    return s.replace("\\", r"\\").replace("'", r"\'").replace("\n", r"\\n")

def add_text_watermark_ffmpeg(
    img_bytes: bytes,
    text: str,
    scale: float,
    position: str = "center",
    opacity: float = 0.9
) -> io.BytesIO:
    _which_ffmpeg()
    if not os.path.exists(FONT_PATH):
        raise RuntimeError(f"Font not found at {FONT_PATH}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_f, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as out_f:
        img_f.write(img_bytes)
        img_f.flush()
        in_img, out_path = img_f.name, out_f.name

    try:
        W = _probe_image_width(in_img)
        font_size = _text_px_from_scale(W, scale)

        margin = 16
        pos = (position or "center").lower()
        if   pos == "tl":
            x_expr, y_expr = f"{margin}", f"{margin}"
        elif pos == "tr":
            x_expr, y_expr = f"(w-text_w)-{margin}", f"{margin}"
        elif pos == "bl":
            x_expr, y_expr = f"{margin}", f"(h-text_h)-{margin}"
        elif pos == "center":
            x_expr, y_expr = "(w-text_w)/2", "(h-text_h)/2"
        else:  # "br" or unknown -> bottom-right
            x_expr, y_expr = f"(w-text_w)-{margin}", f"(h-text_h)-{margin}"

        op = max(0.05, min(1.0, float(opacity)))
        op_shadow = min(1.0, op * 0.7)

        safe_text = _escape_drawtext(text or "")

        vf = (
            f"drawtext=fontfile='{FONT_PATH}':text='{safe_text}':"
            f"fontsize={font_size}:fontcolor=black@{op_shadow}:"
            f"x=({x_expr})+2:y=({y_expr})+2,"
            f"drawtext=fontfile='{FONT_PATH}':text='{safe_text}':"
            f"fontsize={font_size}:fontcolor=white@{op}:"
            f"x={x_expr}:y={y_expr}"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", in_img,
            "-vf", vf,
            "-frames:v", "1",
            "-q:v", "2",
            out_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        with open(out_path, "rb") as f:
            buf = io.BytesIO(f.read())
            buf.seek(0)
            return buf

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}") from e
    finally:
        for p in (in_img, out_path):
            try:
                os.remove(p)
            except:
                pass

def _logo_prompt(text: str, colors: str = "", style: str = "") -> str:
    colors_desc = colors.strip() or "designer's choice within brand-friendly neutrals"
    style_desc  = style.strip()  or "minimal, geometric, high-contrast"
    return (
        "Design a simple, clean LOGO.\n"
        f"- Brand text: \"{text.strip()}\"\n"
        f"- Color palette: {colors_desc}\n"
        f"- Style: {style_desc}\n"
        "- Flat vector look, centered composition, legible typography.\n"
        "- No background, no mockups, no photos or textures.\n"
    )

@app.post("/api/generate-logo")
async def generate_logo(req: Request):
    try:
        data = await req.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    text = (data.get("text") or "").strip()
    colors = (data.get("colors") or "").strip()
    style = (data.get("style") or "").strip()

    if not text:
        raise HTTPException(400, "text is required")

    if len(text) > 80:   text = text[:80]
    if len(colors) > 120: colors = colors[:120]
    if len(style) > 200:  style = style[:200]

    prompt = _logo_prompt(text, colors, style)

    client = get_openai_client()
    try:
        img = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="b64_json",
        )
        data0 = img.data[0]
        b64 = getattr(data0, "b64_json", None) or (data0.get("b64_json") if isinstance(data0, dict) else None)
        if not b64:
            raise HTTPException(502, "OpenAI did not return base64 image data (b64_json).")
        png_bytes = base64.b64decode(b64)
        return StreamingResponse(BytesIO(png_bytes), media_type="image/png", headers={"Cache-Control": "no-store"})
    except Exception as e:
        msg = getattr(e, "message", None) or str(e)
        raise HTTPException(502, f"Logo generation failed: {msg}")

def get_openai_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(500, "OpenAI key not configured (set OPENAI_API_KEY in .env file)")
    return OpenAI(api_key=key)
