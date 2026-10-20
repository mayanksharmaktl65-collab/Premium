"""Premium API - MSD Studio (Mayank Sharma)
Link downloader (yt-dlp) + image/video watermark remover (OpenCV / ffmpeg)
"""
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import cv2
import numpy as np
import yt_dlp
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

MAX_UPLOAD_MB = 100
MAX_DOWNLOAD_MB = 300

app = FastAPI(title="Premium API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


def cleanup(path: str):
    shutil.rmtree(path, ignore_errors=True)


def clamp_pct(v: float) -> float:
    return max(0.0, min(100.0, v))


async def save_upload(file: UploadFile, folder: str) -> str:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"फ़ाइल {MAX_UPLOAD_MB}MB से बड़ी है")
    name = os.path.basename(file.filename or "input")
    path = os.path.join(folder, "in_" + name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def fetch_url(url: str, folder: str) -> str:
    opts = {
        "quiet": True,
        "noplaylist": True,
        "outtmpl": os.path.join(folder, "in_%(id)s.%(ext)s"),
        "format": "best[height<=1080]/best",
        "max_filesize": MAX_DOWNLOAD_MB * 1024 * 1024,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        raise HTTPException(400, "लिंक से वीडियो नहीं मिला: " + str(e)[:200])


@app.get("/")
def root():
    return {"status": "ok", "app": "Premium"}


@app.get("/health")
def health():
    return {"status": "ok"}


class UrlIn(BaseModel):
    url: str


@app.post("/api/info")
def info(body: UrlIn):
    opts = {"quiet": True, "noplaylist": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            d = ydl.extract_info(body.url, download=False)
    except Exception as e:
        raise HTTPException(400, "लिंक से जानकारी नहीं मिल सकी: " + str(e)[:200])
    heights = sorted(
        {f.get("height") for f in d.get("formats", []) if f.get("height")},
        reverse=True,
    )
    return {
        "title": d.get("title"),
        "thumbnail": d.get("thumbnail"),
        "duration": d.get("duration"),
        "uploader": d.get("uploader"),
        "heights": heights,
    }


@app.get("/api/download")
def download(
    bg: BackgroundTasks,
    url: str,
    kind: str = "video",
    quality: str = "720",
    audio_format: str = "mp3",
):
    if audio_format not in ("mp3", "m4a", "wav", "opus", "flac"):
        audio_format = "mp3"
    tmp = tempfile.mkdtemp()
    opts = {
        "quiet": True,
        "noplaylist": True,
        "outtmpl": os.path.join(tmp, "%(title).80s.%(ext)s"),
        "max_filesize": MAX_DOWNLOAD_MB * 1024 * 1024,
    }
    if kind == "audio":
        opts.update(
            {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": audio_format,
                        "preferredquality": "192",
                    }
                ],
            }
        )
    else:
        h = int(quality) if quality.isdigit() else 720
        opts.update(
            {
                "format": f"bestvideo[height<={h}]+bestaudio/best[height<={h}]/best",
                "merge_output_format": "mp4",
            }
        )
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        cleanup(tmp)
        raise HTTPException(400, "डाउनलोड नहीं हो सका: " + str(e)[:200])
    files = os.listdir(tmp)
    if not files:
        cleanup(tmp)
        raise HTTPException(400, "फ़ाइल नहीं बनी (साइज़ बहुत बड़ा हो सकता है)")
    bg.add_task(cleanup, tmp)
    return FileResponse(os.path.join(tmp, files[0]), filename=files[0])


@app.post("/api/watermark/image")
async def watermark_image(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    w: float = Form(...),
    h: float = Form(...),
):
    """x, y, w, h are percentages (0-100) of the image size."""
    tmp = tempfile.mkdtemp()
    bg.add_task(cleanup, tmp)
    path = await save_upload(file, tmp)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "यह इमेज नहीं खुल सकी")
    H, W = img.shape[:2]
    x0 = int(W * clamp_pct(x) / 100)
    y0 = int(H * clamp_pct(y) / 100)
    x1 = int(W * clamp_pct(x + w) / 100)
    y1 = int(H * clamp_pct(y + h) / 100)
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(400, "वाटरमार्क वाला हिस्सा सही से चुनें")
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 255
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    result = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    out = os.path.join(tmp, "premium-clean.png")
    cv2.imwrite(out, result)
    return FileResponse(out, media_type="image/png", filename="premium-clean.png")


@app.post("/api/watermark/video")
async def watermark_video(
    bg: BackgroundTasks,
    x: float = Form(...),
    y: float = Form(...),
    w: float = Form(...),
    h: float = Form(...),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
):
    """Upload a video OR give a link. x, y, w, h are percentages (0-100)."""
    tmp = tempfile.mkdtemp()
    bg.add_task(cleanup, tmp)
    if file is not None:
        src = await save_upload(file, tmp)
    elif url:
        src = fetch_url(url, tmp)
    else:
        raise HTTPException(400, "वीडियो फ़ाइल या लिंक दें")

    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0", src,
            ],
            capture_output=True, text=True, timeout=60,
        )
        W, H = [int(v) for v in probe.stdout.strip().split(",")[:2]]
    except Exception:
        raise HTTPException(400, "यह वीडियो नहीं खुल सका")

    px = max(1, int(W * clamp_pct(x) / 100))
    py = max(1, int(H * clamp_pct(y) / 100))
    pw = int(W * clamp_pct(w) / 100)
    ph = int(H * clamp_pct(h) / 100)
    pw = min(pw, W - px - 1)
    ph = min(ph, H - py - 1)
    if pw < 2 or ph < 2:
        raise HTTPException(400, "वाटरमार्क वाला हिस्सा सही से चुनें")

    out = os.path.join(tmp, "premium-clean.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", f"delogo=x={px}:y={py}:w={pw}:h={ph}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "copy", "-movflags", "+faststart", out,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "वीडियो बहुत लंबा है, छोटा वीडियो आज़माएँ")
    if r.returncode != 0 or not os.path.exists(out):
        raise HTTPException(500, "वाटरमार्क हटाने में दिक्कत आई")
    return FileResponse(out, media_type="video/mp4", filename="premium-clean.mp4")
