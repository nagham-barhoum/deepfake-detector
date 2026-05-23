"""frame_extractor.py — Extract frames from video files"""

import cv2
import tempfile
import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class VideoMeta:
    duration_sec: float
    fps: float
    width: int
    height: int
    codec: str
    total_frames: int

@dataclass
class ExtractedFrame:
    frame_idx: int
    timestamp: float
    is_keyframe: bool
    data: any  # numpy array

def extract_frames(
    video_path: str,
    max_frames: int = 60,
    interval: float = 1.0,
) -> Tuple[List[ExtractedFrame], VideoMeta]:
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps
    fourcc_int   = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec        = "".join([chr((fourcc_int >> 8*i) & 0xFF) for i in range(4)]).strip()

    meta = VideoMeta(
        duration_sec=round(duration_sec, 2),
        fps=round(fps, 2),
        width=width,
        height=height,
        codec=codec,
        total_frames=total_frames,
    )

    # حساب الـ frame indices اللي رح نسحبها
    step_frames = max(1, int(fps * interval))
    indices = list(range(0, total_frames, step_frames))[:max_frames]

    frames: List[ExtractedFrame] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # OpenCV → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(ExtractedFrame(
            frame_idx=idx,
            timestamp=round(idx / fps, 2),
            is_keyframe=(idx % (step_frames * 5) == 0),
            data=frame_rgb,
        ))

    cap.release()
    return frames, meta


def frames_to_temp_images(frames: List[ExtractedFrame]) -> List[str]:
    """يحفظ كل frame كملف temp PNG"""
    from PIL import Image
    tmp_paths = []
    for frame in frames:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img = Image.fromarray(frame.data)
        img.save(tmp.name)
        tmp.close()
        tmp_paths.append(tmp.name)
    return tmp_paths


def cleanup_temp_files(paths: List[str]):
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass