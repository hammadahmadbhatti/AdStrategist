"""The user's own photos and clips in the video ad, with the script written by the local model."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from adstrategist import pipeline
from adstrategist.context import RunConfig
from adstrategist.media import MAX_MEDIA, collect, filename_hint, load
from adstrategist.pipeline import run_pipeline
from adstrategist.sources import vision

from .conftest import FakeProvider

URL = "https://greenleaf.example/"
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def photo(path: Path, size=(1200, 800), color=(60, 130, 90)) -> str:
    Image.new("RGB", size, color).save(path)
    return str(path)


def clip(path: Path, size="640x360", seconds=1.0) -> str:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size={size}:rate=30:duration={seconds}", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def video_run(settings, fetcher, tmp_path, media, **overrides):
    config = RunConfig(url=URL, steps=["video"], platforms=["google"], media=media, **overrides)
    return run_pipeline(config, settings, fetcher=fetcher, run_dir=tmp_path / "run")["results"]["video"]


@pytest.mark.parametrize("name,hint", [
    ("grill-steak_closeup.mp4", "grill steak closeup"), ("IMG_4032.HEIC", ""), ("PXL_20261115_101500.jpg", ""),
    ("Screenshot 2026-11-15 at 10.15.00.png", ""), ("cafe.jpg", "cafe"), ("kitchen (2).jpg", "kitchen"),
    ("video.mov", ""), ("12.jpg", ""), ("Gewürzregal Küche.png", "Gewürzregal Küche"),
])
def test_file_names_become_hints_unless_they_are_camera_names(name, hint):
    assert filename_hint(Path(name)) == hint


def test_collect_keeps_order_expands_folders_and_skips_other_files(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    for name in ("b.jpg", "a.png", "notes.txt", ".hidden.jpg"):
        (folder / name).write_bytes(b"x")
    single = tmp_path / "z-first.mov"
    single.write_bytes(b"x")
    doc = tmp_path / "plan.pdf"
    doc.write_bytes(b"x")
    found, notes = collect([str(single), str(folder), str(doc), str(tmp_path / "missing.jpg")])
    assert [p.name for p in found] == ["z-first.mov", "a.png", "b.jpg"]
    assert any("plan.pdf" in n for n in notes) and any("missing.jpg" in n for n in notes)
    many = [photo(tmp_path / f"p{i}.jpg", size=(300, 300)) for i in range(MAX_MEDIA + 3)]
    found, notes = collect(many)
    assert len(found) == MAX_MEDIA and any(f"first {MAX_MEDIA}" in n for n in notes)


def test_load_rotates_phone_photos_and_rejects_bad_files(tmp_path):
    sideways = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # stored landscape, to be shown rotated to portrait
    Image.new("RGB", (800, 400)).save(sideways, exif=exif)
    item, problem = load(sideways, tmp_path / "work")
    assert problem == "" and item.frame.size == (400, 800) and item.portrait
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    assert load(broken, tmp_path / "work") == (None, "broken.png: the file could not be opened.")
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (50, 50)).save(tiny)
    assert "too small" in load(tiny, tmp_path / "work")[1]


@needs_ffmpeg
def test_own_photos_make_one_scene_each_plus_a_closing_card(settings, fetcher, tmp_path):
    media = [photo(tmp_path / "tea-garden.jpg"), photo(tmp_path / "IMG_0042.jpg", size=(800, 1400)),
             photo(tmp_path / "packing-boxes.png")]
    video = video_run(settings, fetcher, tmp_path, media)  # video mode is "auto" and only Google is selected
    assert video["rendered"], video
    assert len(video["scenes"]) == 4 and video["scenes"][-1]["cta"]
    assert [s["picture"] for s in video["scenes"]] == [
        "Photo: tea-garden.jpg (tea garden)", "Photo: IMG_0042.jpg", "Photo: packing-boxes.png (packing boxes)",
        "Brand colour card"]
    assert video["pictures"] == "3 of your photos and 0 of your clips"
    assert [m["described_by"] for m in video["media"]] == ["file name", "", "file name"]
    assert any("No vision model looked" in n for n in video["notes"])
    assert not video["written_by_model"]
    assert (tmp_path / "run" / "video" / "ad.mp4").stat().st_size > 20_000
    assert not (tmp_path / "run" / "video" / "work").exists()
    assert "Picture: Photo: tea-garden.jpg" in (tmp_path / "run" / "video" / "script.txt").read_text()


@needs_ffmpeg
def test_clips_play_as_footage_and_short_ones_repeat(settings, fetcher, tmp_path):
    media = [clip(tmp_path / "pouring-tea.mp4", "640x360", 1.0), clip(tmp_path / "IMG_7001.mov", "360x640", 4.0),
             photo(tmp_path / "shop.jpg")]
    video = video_run(settings, fetcher, tmp_path, media)
    assert video["rendered"], video
    assert video["pictures"] == "1 of your photos and 2 of your clips"
    assert [m["kind"] for m in video["media"]] == ["video", "video", "image"]
    assert video["scenes"][0]["seconds"] >= 2.6  # longer than the one second clip, so the clip repeats
    assert abs(video["seconds"] - sum(s["seconds"] for s in video["scenes"])) < 1.0
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(tmp_path / "run" / "video" / "ad.mp4")],
                           capture_output=True, text=True).stdout.strip()
    assert probe == "1080,1920"
    assert any("sound muted" in n for n in video["notes"])


@needs_ffmpeg
def test_local_model_writes_the_script_around_what_the_vision_model_saw(settings, fetcher, tmp_path, monkeypatch):
    seen = iter(["A bowl of loose leaf tea on a wooden table. It looks nice.", "A person packing a cardboard box"])
    monkeypatch.setattr(vision.httpx, "get", lambda url, **_: httpx.Response(
        200, json={"models": [{"name": "moondream:latest"}]}, request=httpx.Request("GET", url)))
    asked = []

    def fake_post(url, json=None, **_):
        asked.append(json)
        return httpx.Response(200, json={"message": {"content": next(seen)}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(vision.httpx, "post", fake_post)
    settings.vision_model = "moondream"
    local = FakeProvider({"The pictures are fixed": {"scenes": [
        {"caption": "Tea worth slowing down for", "voiceover": "Loose leaf organic tea, picked on small farms."},
        {"caption": "Jetzt 50% Rabatt", "voiceover": "Half price today only."},
        {"caption": "Start your box", "voiceover": "Order your first GreenLeaf box today."}]}}, model="local-3b")
    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: local)
    media = [photo(tmp_path / "IMG_1.jpg"), photo(tmp_path / "IMG_2.jpg")]
    result = run_pipeline(RunConfig(url=URL, steps=["video"], media=media), settings, fetcher=fetcher,
                          run_dir=tmp_path / "run")
    video = result["results"]["video"]
    assert len(asked) == 2 and asked[0]["model"] == "moondream" and len(asked[0]["messages"][0]["images"]) == 1
    prompt = next(p for p in local.prompts if "The pictures are fixed" in p)
    assert "Shot 1: photo, shows: A bowl of loose leaf tea on a wooden table." in prompt
    assert "Shot 2: photo, shows: A person packing a cardboard box." in prompt
    assert "Write exactly 3 scenes" in prompt
    assert video["written_by_model"] and video["script_model"] == "Fake model"
    assert video["scenes"][0]["caption"] == "Tea worth slowing down for"
    assert "Rabatt" not in video["scenes"][1]["caption"]  # the invented discount is replaced, the shot is kept
    assert video["scenes"][2]["caption"] == "Start your box" and video["scenes"][2]["cta"]
    assert [m["described_by"] for m in video["media"]] == ["vision model", "vision model"]
    assert any("looked at 2 of 2 files" in n for n in video["notes"])
    assert any(c["where"] == "video scene" for c in result["meta"]["rejected_claims"])


@needs_ffmpeg
def test_a_script_that_skips_shots_is_not_trusted(settings, fetcher, tmp_path, monkeypatch):
    lazy = FakeProvider({"The pictures are fixed": {"scenes": [{"caption": "Only one", "voiceover": "Just one line."}]}})
    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: lazy)
    media = [photo(tmp_path / f"shot{i}.jpg") for i in range(3)]
    video = video_run(settings, fetcher, tmp_path, media)
    assert video["rendered"] and not video["written_by_model"] and len(video["scenes"]) == 4
    assert video["scenes"][0]["caption"] != "Only one"


@needs_ffmpeg
def test_unusable_files_are_reported_and_the_rest_still_used(settings, fetcher, tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"nope")
    video = video_run(settings, fetcher, tmp_path, [str(broken), photo(tmp_path / "good.jpg")])
    assert video["rendered"] and len(video["media"]) == 1 and len(video["scenes"]) == 2
    assert any("broken.jpg" in n for n in video["notes"])
    video = run_pipeline(RunConfig(url=URL, steps=["video"], media=[str(broken)]), settings, fetcher=fetcher,
                         run_dir=tmp_path / "run2")["results"]["video"]
    assert video["rendered"] and video["media"] == [] and video["pictures"].endswith("site images")


def test_video_switched_off_wins_over_media(settings, fetcher, tmp_path):
    video = video_run(settings, fetcher, tmp_path, [photo(tmp_path / "a.jpg")], video="no")
    assert not video["rendered"] and "switched off" in video["reason"]


def test_upload_endpoints_store_files_in_order_and_feed_the_run(settings, monkeypatch, tmp_path):
    from adstrategist import server
    monkeypatch.setattr(server, "_settings", lambda: settings)
    captured = {}

    def fake_run(config, _settings, on_event=None, run_dir=None):
        captured["media"] = list(config.media)
        return {"errors": {}}

    monkeypatch.setattr(server, "run_pipeline", fake_run)
    client = TestClient(server.app)
    options = client.get("/api/options").json()["media"]
    assert options["max_files"] == MAX_MEDIA and ".mov" in options["video_types"] and "detail" in options["vision"]

    upload = client.post("/api/uploads").json()["id"]
    assert client.put(f"/api/uploads/{upload}/second shot!.MOV", content=b"clipdata").json()["name"] == "01_second shot.mov"
    assert client.put(f"/api/uploads/{upload}/first.jpg", content=b"jpegdata").json() == {"name": "02_first.jpg", "bytes": 8}
    assert client.put(f"/api/uploads/{upload}/script.sh", content=b"rm -rf").status_code == 415
    assert client.put(f"/api/uploads/{upload}/empty.png", content=b"").status_code == 400
    assert client.put("/api/uploads/0000000000000000/a.jpg", content=b"x").status_code == 404
    assert client.put("/api/uploads/not-an-id/a.jpg", content=b"x").status_code == 404
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 4)
    assert client.put(f"/api/uploads/{upload}/big.png", content=b"12345").status_code == 413
    stored = sorted(p.name for p in (settings.output_dir / "_uploads" / upload).iterdir())
    assert stored == ["01_second shot.mov", "02_first.jpg"]  # rejected files leave nothing behind

    run_id = client.post("/api/runs", json={"url": URL, "upload_id": upload}).json()["id"]
    for _ in range(50):
        if client.get(f"/api/runs/{run_id}").json()["status"] == "done":
            break
        time.sleep(0.05)
    assert [Path(p).name for p in captured["media"]] == ["01_second shot.mov", "02_first.jpg"]
    assert client.post("/api/runs", json={"url": URL, "upload_id": "ffffffffffffffff"}).status_code == 404
    assert client.get("/api/runs").json() == []  # the uploads folder is not listed as a run


@pytest.mark.parametrize("seen,kept", [
    ("ikka logo with anchor and a ship on it.", "A logo with anchor and a ship on it."),
    ("Jars of spices on a table, with a pink sign in the background advertising a sale of Black Spice Days.",
     "Jars of spices on a table, with a sign in the background."),
    ('A storefront with a banner that says "Grand Opening" above the door.', "A storefront with a banner."),
    ("A person packing a cardboard box", "A person packing a cardboard box."),
    ("logo", ""),
])
def test_vision_notes_lose_whatever_the_model_claims_to_have_read(seen, kept):
    from adstrategist.media import neutral
    assert neutral(seen) == kept


@needs_ffmpeg
def test_a_misread_logo_cannot_become_a_brand_in_the_script(settings, fetcher, tmp_path, monkeypatch):
    monkeypatch.setattr(vision.httpx, "get", lambda url, **_: httpx.Response(
        200, json={"models": [{"name": "moondream:latest"}]}, request=httpx.Request("GET", url)))
    monkeypatch.setattr(vision.httpx, "post", lambda url, **_: httpx.Response(
        200, json={"message": {"content": "ikka logo with anchor and a ship on it."}}, request=httpx.Request("POST", url)))
    settings.vision_model = "moondream"
    local = FakeProvider({"The pictures are fixed": {"scenes": [
        {"caption": "ikka, your tea brand", "voiceover": "With ikka you find the perfect cup."},
        {"caption": "Order today", "voiceover": "Order your first box today."}]}})
    monkeypatch.setattr(pipeline, "build_provider", lambda *_a, **_k: local)
    result = run_pipeline(RunConfig(url=URL, steps=["video"], media=[photo(tmp_path / "IMG_9.jpg")]), settings,
                          fetcher=fetcher, run_dir=tmp_path / "run")
    video = result["results"]["video"]
    prompt = next(p for p in local.prompts if "The pictures are fixed" in p)
    assert "ikka" not in prompt and "shows: A logo with anchor" in prompt and "The only brand is GreenLeaf" in prompt
    assert "ikka" not in video["scenes"][0]["caption"].lower() + video["scenes"][0]["voiceover"].lower()
    assert video["media"][0]["description"] == "A logo with anchor and a ship on it."
    assert any("read off a picture" in c["reason"] for c in result["meta"]["rejected_claims"])
