# Fast Watermarker

A lightweight web app built with **FastAPI** and **Bootstrap 5** to watermark images quickly.  
Supports **text watermarks**, **logo watermarks**, and **batch uploads** (multiple images at once, returned as a `.zip`).

---

## Features

- Upload **one or more images** (up to 20 at a time).
- Add a **text watermark** (e.g., "© mybrand").
- Add a **logo watermark** (PNG recommended, transparency supported).
- Control **logo scale** with a slider.
- Smart handling of transparent logos.
- Client-side loading indicators (spinner + status banners).
- Processed results: downloadable `.zip`.

---

## Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/), [Hypercorn](https://pypi.org/project/Hypercorn/).
- **Frontend**: Bootstrap 5, vanilla JS.
- **Imaging**: [Pillow (PIL)](https://python-pillow.org/).
- **Templating**: Jinja2.
- **File uploads**: `python-multipart`.