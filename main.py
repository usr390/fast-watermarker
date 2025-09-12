from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import File, UploadFile, Form

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
      <head>
        <title>Fast Watermarker</title>
      </head>
      <body>
        <h1>Upload an Image</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept="image/*">
          <input type="text" name="text" placeholder="Watermark text">
          <button type="submit">Upload</button>
        </form>
      </body>
    </html>
    """
@app.post("/upload")
async def upload(file: UploadFile = File(...), text: str = Form(...)):
    return {"filename": file.filename, "text": text}