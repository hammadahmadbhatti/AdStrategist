"""The user's own photos and video clips for the video ad: finding, opening and describing them."""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
MAX_MEDIA = 8

# Camera and screenshot file names carry no meaning ("IMG_4032", "PXL_2026...", "Screenshot 2026-...").
_CAMERA_NAME = re.compile(r"^(img|dsc|dscn|pxl|vid|mov|mvi|gopr|photo|image|video|clip|screenshot|whatsapp|"
                          r"bildschirmfoto|screen recording|untitled)?[\s_\-]*(?:[\d\s_\-.:()]|\bat\b|\bum\b)*$", re.I)


@dataclass
class MediaItem:
    path: Path
    kind: str                 # "image" or "video"
    frame: Image.Image        # the photo itself, or a representative frame of the clip
    duration: float = 0.0     # clips only
    description: str = ""
    described_by: str = ""    # "vision model", "file name" or ""
    raw_description: str = "" # what the vision model said before names and readings were stripped

    @property
    def portrait(self) -> bool:
        return self.frame.width / self.frame.height <= 0.8

    @property
    def name(self) -> str:
        """The file name as the user knows it (uploads are stored with an order prefix like '01_')."""
        return re.sub(r"^\d{2}_", "", self.path.name) if self.path.parent.parent.name == "_uploads" else self.path.name

    def label(self) -> str:
        return f"{'Clip' if self.kind == 'video' else 'Photo'}: {self.name}"


def collect(paths: list[str]) -> tuple[list[Path], list[str]]:
    """Files to use, in the order given. Folders are expanded alphabetically."""
    found: list[Path] = []
    notes: list[str] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            entries = sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith("."))
        elif path.is_file():
            entries = [path]
        else:
            notes.append(f"Not found: {raw}")
            continue
        for entry in entries:
            if entry.suffix.lower() in IMAGE_EXT | VIDEO_EXT:
                if entry not in found:
                    found.append(entry)
            elif path.is_file():
                notes.append(f"Skipped {entry.name}: only photos ({', '.join(sorted(IMAGE_EXT))}) and clips "
                             f"({', '.join(sorted(VIDEO_EXT))}) can be used.")
    if len(found) > MAX_MEDIA:
        notes.append(f"{len(found)} files were given. The first {MAX_MEDIA} are used: a short ad works best "
                     "with a few strong shots.")
        found = found[:MAX_MEDIA]
    return found, notes


def filename_hint(path: Path) -> str:
    """'grill-steak_closeup.mp4' -> 'grill steak closeup'. Camera-style names give nothing."""
    stem = path.stem
    if _CAMERA_NAME.match(stem):
        return ""
    words = re.sub(r"[_\-.]+", " ", stem)
    words = re.sub(r"\s*\(?\b\d+\)?$", "", words).strip()  # trailing counters: "kitchen 2", "kitchen (3)"
    return words if len(words) >= 3 else ""


_READS = re.compile(r"\b(?:advertising|that (?:says|reads)|which (?:says|reads)|saying|reading|"
                    r"with the (?:words?|text|name)|labell?ed|titled|called|named)\b", re.I)


def neutral(description: str) -> str:
    """Strip what a vision model claims to have *read*. Small vision models misread logos and signs
    ("ikka logo"), and a script that repeats the misreading advertises a brand that does not exist.
    What the picture shows is kept; names, quoted text and anything after "that says" are dropped."""
    text = re.sub(r"[\"“”‘’][^\"“”‘’]{1,60}[\"“”‘’]", "", description)
    text = _READS.split(text)[0]
    text = re.sub(r"\b(?:(?:a|an|the)\s+)?[\w'’-]+ (logo|sign|label|banner|brand)\b", r"a \1", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")
    return (text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")) if len(text) >= 8 else ""


def _clip_duration(path: Path) -> float:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=nw=1:nk=1", str(path)], capture_output=True, text=True, timeout=30).stdout
        return float(out.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def load(path: Path, work: Path) -> tuple[MediaItem | None, str]:
    """Open one file. Returns (item, '') or (None, why it cannot be used)."""
    suffix = path.suffix.lower()
    work.mkdir(parents=True, exist_ok=True)
    try:
        if suffix in VIDEO_EXT:
            if not shutil.which("ffmpeg"):
                return None, f"{path.name}: ffmpeg is needed to use video clips."
            duration = _clip_duration(path)
            if duration < 0.3:
                return None, f"{path.name}: the clip could not be read or is shorter than a third of a second."
            still = work / f"{path.stem}_frame.jpg"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{min(1.0, duration / 2):.2f}", "-i",
                            str(path), "-frames:v", "1", "-q:v", "3", str(still)], capture_output=True, timeout=60)
            frame = Image.open(still)
            frame.load()
            return MediaItem(path, "video", frame.convert("RGB"), duration), ""
        source = path
        if suffix in (".heic", ".heif"):  # iPhone photos: Pillow cannot read them, macOS can convert them
            if not shutil.which("sips"):
                return None, f"{path.name}: HEIC photos can only be converted on macOS. Export it as JPEG."
            source = work / f"{path.stem}.jpg"
            subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(source)],
                           capture_output=True, timeout=60)
        picture = Image.open(source)
        picture.load()
        picture = ImageOps.exif_transpose(picture).convert("RGB")  # phone photos store rotation separately
        if min(picture.size) < 200:
            return None, f"{path.name}: the photo is too small (under 200 pixels)."
        return MediaItem(path, "image", picture), ""
    except Exception:
        return None, f"{path.name}: the file could not be opened."


def describe_all(items: list[MediaItem], host: str, model: str, vision_ready: bool, emit) -> None:
    """Fill in what each item shows: from the local vision model when there is one, else the file name."""
    from .sources.vision import describe

    for index, item in enumerate(items, 1):
        if vision_ready:
            emit("info", f"Looking at {item.path.name} ({index} of {len(items)}) with {model}")
            text = describe(host, model, item.frame)
            if neutral(text):
                item.raw_description, item.description, item.described_by = text, neutral(text), "vision model"
                continue
        hint = filename_hint(Path(item.name))
        if hint:
            item.description, item.described_by = hint, "file name"
