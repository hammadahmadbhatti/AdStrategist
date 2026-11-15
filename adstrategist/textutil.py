"""Small text helpers shared by the steps. No model involved."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from urllib.parse import urlparse

STOP_EN = set(
    "a an and are as at be but by for from has have how i if in into is it its more most my no not of on or "
    "our so than that the their them these they this to up us was we what when where which who why will with "
    "you your all any can do does get got just new now one out over also about been here there".split()
)
STOP_DE = set(
    "aber alle als am an auch auf aus bei bin bis das dass dem den der des die dir du durch ein eine einem "
    "einen einer eines er es für hat haben ich ihr ihre im in ist ja kann mehr mein mit nach nicht noch nur "
    "oder ohne sein sich sie sind so über um und uns unser unsere von vor was wenn wer wie wir wird zu zum zur "
    "jetzt hier ihnen ihren ihrem sowie bzw".split()
)
STOP = STOP_EN | STOP_DE | {"home", "startseite", "welcome", "willkommen", "www", "com", "de", "inc", "gmbh", "ltd"}

# Words that appear on every website and say nothing about the product.
WEB_STOP = set(
    "discover learn more get started start free contact about home blog login sign signup pricing features feature "
    "solutions solution resources products product company customers customer support help read see view try today "
    "best top world team teams business click here now shop buy order menu search account cookie cookies privacy "
    "terms rights reserved copyright insights overview welcome page site website online way thing things everything "
    "need needs make makes made work works place one use using used people time day year years "
    "credit card trial per unlimited pro plan plans month monthly annual annually user users million millions "
    "billion trusted easy simple better great"
    "mehr erfahren kontakt startseite über anmelden preise funktionen lösungen produkte unternehmen kunden hilfe "
    "lesen ansehen heute jetzt alle rechte vorbehalten datenschutz impressum seite webseite "
    "kaufen bestellen shop warenkorb vorgeschlagenen vorgeschlagene empfohlen neu angebote entdecken".split()
)

LANGUAGE_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "nl": "Dutch",
    "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "sv": "Swedish", "da": "Danish", "ms": "Malay",
}

_WORD = re.compile(r"[^\W\d_][\w'’-]*", re.UNICODE)
_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu"}


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokens(text: str | None) -> list[str]:
    return [w.lower().strip("'’-") for w in _WORD.findall(text or "")]


def content_tokens(text: str | None) -> list[str]:
    return [t for t in tokens(text) if t not in STOP and len(t) > 2]


def truncate(text: str, limit: int) -> str:
    """Cut at a word boundary so the result is never longer than limit."""
    text = clean(text)
    if len(text) <= limit:
        return text
    cut = text[: limit + 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut[:limit].rstrip(" ,;:-–|")


def truncate_sentence(text: str, limit: int) -> str:
    """Keep whole sentences when they fit, otherwise fall back to a word-boundary cut."""
    text = clean(text)
    if len(text) <= limit:
        return text
    kept = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(kept) + len(sentence) + 1 > limit:
            break
        kept = f"{kept} {sentence}".strip()
    return kept if len(kept) >= limit * 0.4 else truncate(text, limit)


def word_caps(text: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in clean(text).split(" "))


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "run"


def host_of(url: str) -> str:
    host = (urlparse(url if "//" in url else "https://" + url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def registrable_domain(url: str) -> str:
    """Best-effort eTLD+1 without a public-suffix list (handles co.uk style suffixes)."""
    parts = host_of(url).split(".")
    if len(parts) <= 2:
        return ".".join(parts)
    if parts[-2] in _SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def dedupe(items, key=lambda x: x.lower()) -> list:
    seen, out = set(), []
    for item in items:
        k = key(item)
        if k and k not in seen:
            seen.add(k)
            out.append(item)
    return out


def _grams(text: str, sizes=(2, 3)):
    for chunk in re.split(r"[.,;:!?|()\[\]{}\"“”•·–—/\n]+", text or ""):
        toks = tokens(chunk)
        for n in sizes:
            for i in range(len(toks) - n + 1):
                gram = toks[i : i + n]
                if gram[0] in STOP or gram[-1] in STOP or len(gram[0]) < 3 or len(gram[-1]) < 3:
                    continue
                if all(t in STOP or t in WEB_STOP for t in gram):
                    continue
                yield " ".join(gram)


def top_phrases(weighted_texts: list[tuple[str, float]], exclude: set[str], limit: int = 8, body: str = "") -> list[str]:
    """Rank 2 to 3 word phrases for rule-based seed keywords.

    A phrase scores for appearing in the title and headings, and scores more for recurring in the
    body text. Slogans appear once; what the product actually is gets repeated all over the page.
    """
    body_counts = Counter(_grams(body))
    scores: Counter[str] = Counter()
    for text, weight in weighted_texts:
        for gram in _grams(text):
            scores[gram] += weight
    for gram, count in body_counts.items():
        if count >= 3 or gram in scores:
            scores[gram] += min(count, 8)
    chosen: list[str] = []
    for phrase, _ in scores.most_common(80):
        if any(t in exclude for t in phrase.split()):
            continue
        if any(phrase in other or other in phrase for other in chosen):
            continue  # keep the higher scoring of two overlapping phrases
        chosen.append(phrase)
        if len(chosen) >= limit:
            break
    return chosen
