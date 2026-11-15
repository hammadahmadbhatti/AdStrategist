"""Step 9: a vertical video ad built locally.

Pictures come from the user's own photos and clips when given, otherwise from the site's images.
The local model writes the script, a local vision model can look at the pictures first, and
Pillow plus ffmpeg do the rendering. Nothing is sent to a paid service.
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..context import Context
from ..media import MediaItem, collect, describe_all, load
from ..sources.vision import vision_status
from ..textutil import content_tokens, truncate
from .campaigns import pack_for

W, H, FPS = 1080, 1920, 30
FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
         "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
VOICES = {"de": ["Anna", "Petra", "Markus"], "en": ["Samantha", "Daniel", "Karen", "Moira", "Alex"],
          "fr": ["Thomas", "Amelie"], "es": ["Monica", "Paulina"], "it": ["Alice"], "nl": ["Xander"],
          "pt": ["Luciana", "Joana"], "tr": ["Yelda"], "pl": ["Zosia"], "sv": ["Alva"], "ms": ["Amira"]}

SCHEMA = {
    "type": "object",
    "properties": {"scenes": {"type": "array", "items": {
        "type": "object",
        "properties": {"caption": {"type": "string", "description": "on-screen text, at most 8 words"},
                       "voiceover": {"type": "string", "description": "spoken sentence, at most 18 words"}},
        "required": ["caption", "voiceover"], "additionalProperties": False}}},
    "required": ["scenes"], "additionalProperties": False,
}


def should_render(ctx: Context, has_media: bool = False) -> tuple[bool, str]:
    mode = ctx.config.video
    if mode == "no":
        return False, "Video was switched off for this run."
    if mode == "yes" or has_media:  # handing over photos or clips is a request for a video
        return True, ""
    video_platforms = {"meta", "tiktok"} & set(ctx.config.platforms)
    if not video_platforms:
        return False, "Only Google Search was selected, and search ads are text."
    budget = ctx.results.get("budget")
    if budget and not ({row["platform"] for row in budget["plan"]} & video_platforms):
        return False, "The budget plan funds no video platform (Meta or TikTok), so a video would go unused."
    return True, ""


def template_scenes(brief: dict, lang: str, shots: int = 0) -> list[dict]:
    """Rule-based script. With `shots` it has exactly that many scenes plus the closing one; shots
    beyond the available selling points run without a caption rather than repeating one."""
    pack = pack_for(lang)
    category = brief["category"] if lang == "de" else brief["category"].lower()  # German nouns keep capitals
    hook = pack["hook"].format(category=category, brand=brief["brand"])
    scenes = [{"caption": hook, "voiceover": hook}]
    for usp in brief.get("usps", [])[: (shots - 1 if shots else 3)]:
        scenes.append({"caption": truncate(usp, 60), "voiceover": usp.rstrip(".") + "."})
    if len(scenes) == 1 and brief.get("one_liner") and shots != 1:
        scenes.append({"caption": truncate(brief["one_liner"], 60), "voiceover": brief["one_liner"]})
    while shots and len(scenes) < shots:
        scenes.append({"caption": "", "voiceover": ""})
    closing = pack["cta_line"][brief["business_type"]]
    scenes.append({"caption": closing.rstrip("."), "voiceover": f"{brief['brand']}. {closing}", "cta": True})
    return scenes


def _facts(brief: dict) -> str:
    return (f"brand {brief['brand']}; {brief['one_liner']}; selling points: {brief.get('usps')}; "
            f"offers on page: {brief.get('offers')}")


def write_script(ctx: Context, media: list[MediaItem]) -> tuple[list[dict], bool]:
    """The script, written by the local model when one is available. With the user's own media there
    is one scene per shot, in the user's order, written to fit what each shot shows."""
    brief, role = ctx.brief, ctx.settings.video_script_role
    role = role if role in ("research", "writing") else "research"
    shots = len(media)
    fallback = template_scenes(brief, ctx.lang, shots)
    if not ctx.llm.enabled(role):
        return fallback, False
    rules = ("caption: on-screen text, at most 8 words. voiceover: one spoken sentence, at most 18 words.\n"
             "Only claim what the facts support.\n\n")
    if media:
        listing = "\n".join(f"Shot {i}: {'video clip' if m.kind == 'video' else 'photo'}"
                             f"{', shows: ' + m.description if m.description else ''}"
                             for i, m in enumerate(media, 1))
        prompt = (f"Write the script for a vertical video ad in {ctx.language_name}. The pictures are fixed: "
                  f"{shots} shots in this order, then a closing scene.\n{listing}\n\n"
                  f"The shot notes come from an automatic tool. The only brand is {brief['brand']}: never take "
                  "a name, a price or wording from the shot notes.\n"
                  f"Write exactly {shots + 1} scenes: one for each shot, matching what that shot shows, then the "
                  "call to action. Scene 1 must hook the viewer.\n" + rules + f"Facts: {_facts(brief)}")
        wanted = shots + 1
    else:
        prompt = (f"Write a 5 scene script for a 20 second vertical video ad in {ctx.language_name}.\n"
                  "Scene 1 hooks the viewer, scenes 2 to 4 each make one point, scene 5 is the call to action.\n"
                  + rules + f"Facts: {_facts(brief)}")
        wanted = 5
    answer = ctx.llm.ask_json("video script", prompt, SCHEMA, role=role, max_tokens=140 * wanted + 100,
                              repeat={"scenes": wanted})
    drafts = [s for s in (answer or {}).get("scenes", []) if s["caption"] and s["voiceover"]]
    if not media:
        usable = [s for s in drafts if ctx.supported(f"{s['caption']}. {s['voiceover']}", "video scene")][:6]
        if len(usable) < 3:
            return fallback, False
        scenes = [{"caption": truncate(s["caption"], 70), "voiceover": truncate(s["voiceover"], 200)} for s in usable]
        scenes[-1]["cta"] = True
        return scenes, True
    if len(drafts) < shots:  # the model did not cover every shot: its lines would land on the wrong pictures
        return fallback, False
    # Words the vision model claimed to read off a picture (and that were stripped from its note) must
    # not come back through the script unless the product page really uses them.
    page = set(content_tokens(" ".join(f"{p.title} {p.text}" for p in [ctx.own_page, *ctx.sub_pages] if p)))
    misread = {t for m in media for t in set(content_tokens(m.raw_description)) - set(content_tokens(m.description))
               if t not in page}
    scenes = []
    for i in range(shots + 1):
        draft = drafts[i] if i < len(drafts) else None
        line = f"{draft['caption']}. {draft['voiceover']}" if draft else ""
        echoed = misread & set(content_tokens(line))
        if echoed:
            ctx.rejected_claims.append({"where": "video scene", "text": line[:160],
                                        "reason": f"“{sorted(echoed)[0]}” was read off a picture by the vision "
                                                  "model and is not on the page"})
        if draft and not echoed and ctx.supported(line, "video scene"):
            scenes.append({"caption": truncate(draft["caption"], 70), "voiceover": truncate(draft["voiceover"], 200)})
        else:
            spare = dict(fallback[i])  # keep the shot, use the rule-based line for it
            if spare["caption"] and spare["caption"] in {x["caption"] for x in scenes} and i < shots:
                spare = {"caption": "", "voiceover": ""}  # never say the same thing twice: the shot plays silently
            scenes.append(spare)
    scenes[-1]["cta"] = True
    return scenes, True


def _font(size: int) -> ImageFont.ImageFont:
    for path in FONTS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    return lines + ([current] if current else [])


def _hex(color: str, default=(31, 58, 95)) -> tuple[int, int, int]:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", (color or "").strip())
    if not match:
        return default
    value = match.group(1)
    rgb = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    return default if sum(rgb) > 600 else rgb  # near-white brand colours make unreadable slides


def background(image: Image.Image | None, brand_rgb: tuple[int, int, int]) -> Image.Image:
    if image is None:
        canvas = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(canvas)
        for y in range(H):
            f = 1 - 0.55 * y / H
            draw.line([(0, y), (W, y)], fill=tuple(int(c * f) for c in brand_rgb))
        return canvas
    scale = max(W / image.width, H / image.height)
    resized = image.resize((round(image.width * scale) + 1, round(image.height * scale) + 1), Image.LANCZOS)
    left, top = (resized.width - W) // 2, (resized.height - H) // 2
    canvas = resized.crop((left, top, left + W, top + H)).convert("RGB")
    shade = Image.new("RGB", (W, H), (0, 0, 0))
    if image.width / image.height <= 0.8:
        return Image.blend(canvas, shade, 0.35)  # already portrait: fill the frame
    # Landscape or square: cropping it to 9:16 would cut most of it away. Show the whole image as a
    # card on top of a blurred, darkened copy of itself instead.
    canvas = Image.blend(canvas.filter(ImageFilter.GaussianBlur(45)), shade, 0.55)
    fit = min((W - 100) / image.width, H * 0.40 / image.height)
    card = image.resize((round(image.width * fit), round(image.height * fit)), Image.LANCZOS).convert("RGB")
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, card.width, card.height], radius=36, fill=255)
    canvas.paste(card, ((W - card.width) // 2, int(H * 0.32) - card.height // 2), mask)
    return canvas


def overlay(caption: str, brand: str, domain: str, brand_rgb: tuple[int, int, int], is_cta: bool) -> Image.Image:
    """Transparent layer with the caption. Kept separate so the background can zoom while text stays still."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    small = _font(46)
    draw.text((W // 2, 150), brand, font=small, fill=(255, 255, 255, 235), anchor="mm",
              stroke_width=3, stroke_fill=(0, 0, 0, 160))
    if not caption:
        return layer
    font = _font(84 if len(caption) <= 40 else 68)
    lines = _wrap(draw, caption, font, W - 200)[:5]
    line_height = int(font.size * 1.25)
    block = line_height * len(lines)
    top = int(H * 0.68) - block // 2
    widest = max(draw.textlength(line, font=font) for line in lines)
    pad = 48
    draw.rounded_rectangle([(W - widest) / 2 - pad, top - pad, (W + widest) / 2 + pad, top + block + pad // 2],
                           radius=36, fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        draw.text((W // 2, top + i * line_height + line_height // 2), line, font=font, fill=(255, 255, 255, 255),
                  anchor="mm")
    if is_cta:
        button = _font(58)
        width = draw.textlength(domain, font=button) + 120
        y = top + block + 150
        draw.rounded_rectangle([(W - width) / 2, y - 62, (W + width) / 2, y + 62], radius=62,
                               fill=(*brand_rgb, 255), outline=(255, 255, 255, 255), width=4)
        draw.text((W // 2, y), domain, font=button, fill=(255, 255, 255, 255), anchor="mm")
    return layer


def pick_voice(lang: str) -> str | None:
    if not shutil.which("say"):
        return None
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    installed = {m.group(1).strip(): m.group(2) for m in
                 (re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line) for line in listing.splitlines()) if m}
    for name in VOICES.get(lang, []):
        if installed.get(name, "").startswith(lang):
            return name
    return None


def _run(cmd: list[str], timeout: int = 180) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _duration(path: Path) -> float:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=nw=1:nk=1", str(path)], capture_output=True, text=True, timeout=30).stdout
        return float(out.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _clip_graph(portrait: bool) -> str:
    """ffmpeg filters that fit a clip into the 9:16 frame, the same way photos are fitted."""
    if portrait:
        return (f"[0:v]fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                "eq=brightness=-0.06,setsar=1[bg];")
    return (f"[0:v]fps={FPS},split[a][b];"
            f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=40:4,"
            "eq=brightness=-0.28[blur];"
            f"[b]scale={W - 100}:{int(H * 0.40)}:force_original_aspect_ratio=decrease[fg];"
            f"[blur][fg]overlay=(W-w)/2:{int(H * 0.32)}-h/2,setsar=1[bg];")


def render(scenes: list[dict], visuals: list, brand: str, domain: str, brand_rgb, voice: str | None,
           folder: Path) -> tuple[Path | None, float]:
    """visuals[i] is a MediaItem, a PIL image, or None (brand colour card) for scene i."""
    work = folder / "work"
    work.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, scene in enumerate(scenes):
        bg, ov, audio, clip = work / f"bg{i}.png", work / f"ov{i}.png", work / f"voice{i}.aiff", work / f"clip{i}.mp4"
        visual = visuals[i]
        is_clip = isinstance(visual, MediaItem) and visual.kind == "video"
        still = visual.frame if isinstance(visual, MediaItem) else visual
        layer = overlay(scene["caption"], brand, domain, brand_rgb, bool(scene.get("cta")))
        layer.save(ov)
        if i == 0:
            Image.alpha_composite(background(still, brand_rgb).convert("RGBA"), layer).convert("RGB").save(
                folder / "poster.jpg", quality=88)
        spoken = bool(voice and scene["voiceover"]) and _run(
            ["say", "-v", voice, "-r", "180", "-o", str(audio), "--", scene["voiceover"]])
        if spoken:
            seconds = max(2.6, _duration(audio) + 0.5)
        elif scene["caption"]:
            seconds = max(2.6, 0.38 * len(scene["caption"].split()) + 1.6)
        else:
            seconds = min(max(visual.duration, 2.0), 3.5) if is_clip else 2.6
        scene["seconds"] = round(seconds, 2)
        frames = int(seconds * FPS)
        tail = "[bg][1:v]overlay=0:0,format=yuv420p[v];[2:a]aresample=44100,apad[a]"
        if is_clip:  # the clip plays as footage; it loops if it is shorter than its line
            source = ["-stream_loop", "-1", "-i", str(visual.path)]
            graph = _clip_graph(visual.portrait) + tail
        else:
            background(still, brand_rgb).save(bg)
            zoom = f"1+0.08*on/{frames}" if i % 2 == 0 else f"1.08-0.08*on/{frames}"
            source = ["-i", str(bg)]
            graph = (f"[0:v]scale={int(W*1.5)}:{int(H*1.5)},zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':"
                     f"y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS}[bg];" + tail)
        sound = ["-i", str(audio)] if spoken else ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
        ok = _run(["ffmpeg", "-y", "-loglevel", "error", *source, "-i", str(ov), *sound,
                   "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-t", f"{seconds:.2f}",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", str(FPS),
                   "-c:a", "aac", "-b:a", "128k", "-ac", "2", str(clip)], timeout=300)
        if not ok:
            return None, 0.0
        clips.append(clip)
    listing = work / "clips.txt"
    listing.write_text("".join(f"file '{c.name}'\n" for c in clips), encoding="utf-8")
    final = folder / "ad.mp4"
    if not _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(final)]):
        return None, 0.0
    shutil.rmtree(work, ignore_errors=True)
    return final, _duration(final)


def _srt(scenes: list[dict]) -> str:
    def stamp(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    out, clock = [], 0.0
    for i, scene in enumerate(scenes, 1):
        end = clock + scene.get("seconds", 3.0)
        if scene["voiceover"]:
            out.append(f"{len(out) + 1}\n{stamp(clock)} --> {stamp(end)}\n{scene['voiceover']}\n")
        clock = end
    return "\n".join(out)


def _site_images(ctx: Context, wanted: int) -> list[Image.Image]:
    images = []
    for url in ctx.brief.get("images", [])[:8]:
        raw = ctx.fetcher.download(url)
        if not raw:
            continue
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except Exception:
            continue
        if min(image.size) >= 300:
            images.append(image.convert("RGB"))
        if len(images) >= wanted:
            break
    return images


def run(ctx: Context) -> dict:
    paths, notes = collect(ctx.config.media)
    render_it, reason = should_render(ctx, has_media=bool(paths))
    if not render_it:
        return {"rendered": False, "reason": reason, "scenes": [], "files": [], "notes": notes}
    brief = ctx.brief
    folder = ctx.run_dir / "video"
    folder.mkdir(parents=True, exist_ok=True)

    media: list[MediaItem] = []
    for path in paths:
        item, problem = load(path, folder / "work")
        if item:
            media.append(item)
        else:
            notes.append(problem)
    if media:
        vision = vision_status(ctx.settings.ollama_host, ctx.settings.vision_model)
        describe_all(media, ctx.settings.ollama_host, ctx.settings.vision_model, vision["ready"], ctx.emit)
        seen = sum(1 for m in media if m.described_by == "vision model")
        if seen:
            notes.append(f"The local vision model {ctx.settings.vision_model} looked at {seen} of {len(media)} "
                         "files, and the script was written to match what it saw.")
        else:
            notes.append("No vision model looked at your files, so the script follows their order and file "
                         "names only. " + ("" if vision["ready"] else vision["detail"]))

    scenes, written_by_model = write_script(ctx, media)
    if media:
        visuals: list = [*media, None]  # the closing scene is a clean brand card
        for scene, item in zip(scenes, media):
            scene["picture"] = item.label() + (f" ({item.description})" if item.description else "")
        scenes[-1]["picture"] = "Brand colour card"
        notes.append("Clips play as footage with their own sound muted, so the voice-over is clear. A clip "
                     "shorter than its line repeats; a longer one is cut to fit.")
    else:
        site = _site_images(ctx, len(scenes))
        visuals = [site[i % len(site)] if site else None for i in range(len(scenes))]
        for scene in scenes:
            scene["picture"] = "Image from the site" if site else "Brand colour card"
        notes.insert(0, "This is a slideshow made from your site's own images, with captions and a system "
                        "voice. Add your own photos or clips for a stronger ad.")
        if not site:
            notes.append("No usable images were found on the page, so the slides use a colour background.")

    (folder / "script.txt").write_text(
        "\n\n".join(f"Scene {i}\nPicture: {s.get('picture', '')}\nOn screen: {s['caption']}\nVoice: {s['voiceover']}"
                    for i, s in enumerate(scenes, 1)), encoding="utf-8")
    files = ["video/script.txt"]
    base = {"scenes": scenes, "files": files, "notes": notes, "written_by_model": written_by_model,
            "script_model": ctx.llm.providers[ctx.settings.video_script_role
                                              if ctx.settings.video_script_role in ("research", "writing")
                                              else "research"].label if written_by_model else "",
            "media": [{"file": m.name, "kind": m.kind, "description": m.description,
                       "described_by": m.described_by} for m in media]}
    if not shutil.which("ffmpeg"):
        shutil.rmtree(folder / "work", ignore_errors=True)
        return {"rendered": False, "reason": "ffmpeg is not installed, so only the script was written. "
                "Install it with: brew install ffmpeg", **base}

    voice = pick_voice(ctx.lang)
    if not voice:
        notes.append(f"No system voice for '{ctx.lang}' was found, so the video has captions only.")
    ctx.emit("info", f"Rendering {len(scenes)} scenes with ffmpeg")
    brand_rgb = _hex(ctx.own_page.theme_color if ctx.own_page else "")
    final, seconds = render(scenes, visuals, brief["brand"], brief["domain"], brand_rgb, voice, folder)
    if not final:
        shutil.rmtree(folder / "work", ignore_errors=True)
        return {"rendered": False, "reason": "ffmpeg failed while rendering. The script was still written.", **base}
    (folder / "captions.srt").write_text(_srt(scenes), encoding="utf-8")
    files += ["video/ad.mp4", "video/poster.jpg", "video/captions.srt"]
    clips = sum(1 for m in media if m.kind == "video")
    pictures = (f"{len(media) - clips} of your photos and {clips} of your clips" if media else
                f"{sum(1 for v in visuals if v is not None)} site images")
    return {"rendered": True, "reason": "", "file": "video/ad.mp4", "poster": "video/poster.jpg",
            "seconds": round(seconds, 1), "format": f"{W}x{H} vertical, H.264", "voice": voice or "",
            "pictures": pictures, **base}
