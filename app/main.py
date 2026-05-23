# """
# main.py — FastAPI Application Entry Point

# Run:
#     uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Swagger docs:
#     http://localhost:8000/docs
# """

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# from api.v1.images import router as images_router
# from api.v1.videos import router as videos_router


# # ─── App ─────────────────────────────────────────────────────────────────────

# app = FastAPI(
#     title       = "Deepfake Detector API",
#     description = "Multi-layer forensic AI image and video detection",
#     version     = "2.0.0",
#     docs_url    = "/docs",
#     redoc_url   = "/redoc",
# )


# # ─── CORS ─────────────────────────────────────────────────────────────────────

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins    = ["http://localhost:3000", "http://localhost:5173"],  # frontend URLs
#     allow_credentials= True,
#     allow_methods    = ["*"],
#     allow_headers    = ["*"],
# )


# # ─── Routers ──────────────────────────────────────────────────────────────────

# API_PREFIX = "/api/v1"

# app.include_router(images_router, prefix=API_PREFIX)
# app.include_router(videos_router, prefix=API_PREFIX)


# # ─── Health ───────────────────────────────────────────────────────────────────

# @app.get("/health", tags=["System"])
# async def health() -> dict:
#     return {"status": "ok", "version": "2.0.0"}


# @app.get("/", tags=["System"])
# async def root() -> dict:
#     return {
#         "message" : "Deepfake Detector API",
#         "docs"    : "/docs",
#         "version" : "2.0.0",
#         "endpoints": {
#             "image": "POST /api/v1/analyze/image",
#             "video": "POST /api/v1/analyze/video",
#         },
#     }
"""
main.py — FastAPI Application Entry Point

Run (from backend/app/):
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Browser UI:  http://localhost:8000/
Swagger:     http://localhost:8000/docs
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.v1.images import router as images_router
from api.v1.videos import router as videos_router


# ─── Paths ────────────────────────────────────────────────────────────────────

TEMPLATES = Jinja2Templates(directory=APP_DIR / "frontend" / "templates")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Deepfake Detector API",
    description = "Multi-layer forensic AI image and video detection",
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.mount(
    "/static",
    StaticFiles(directory=APP_DIR / "frontend" / "static"),
    name="static",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── API routers ──────────────────────────────────────────────────────────────

app.include_router(images_router, prefix="/api/v1")
app.include_router(videos_router, prefix="/api/v1")


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


# ─── Browser UI — single route, no duplicates ─────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def index(request: Request):
    """Serves the browser UI. All analysis calls go to /api/v1/* directly."""
    return TEMPLATES.TemplateResponse("index.html", {"request": request})