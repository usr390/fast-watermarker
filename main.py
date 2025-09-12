from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi import File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...), text: str = Form(...)):
    fake_download_url = "/static/downloads/test.png"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "success": True,
            "filename": file.filename,
            "wm_text": text,
            "download_url": fake_download_url,
        },
    )