"""Claim guard: ad copy may only state what the product page supports.

Small models invent discounts, numbers and superlatives. Every line of copy goes through
`unsupported()` and is dropped when it promises something the page does not show.
"""
from __future__ import annotations

import re

# Offer wording, keyed by the offer signal that must have been detected on the page to allow it.
OFFER_WORDS = {
    "discount": r"rabatt|% ?off|\bdiscounts?\b|gutschein|coupon|\bsale\b|reduziert|\bsparen\b|\bsave \d",
    "free_shipping": r"versandkostenfrei|kostenlose[rn]? versand|gratis versand|free shipping|free delivery",
    "free_trial": r"kostenlos testen|gratis testen|free trial|try (?:it )?(?:for )?free",
    "guarantee": r"garantie|guarantee|money[- ]back|geld[- ]zurück",
    "free_quote": r"kostenlose[sn]? angebot|free quote|free estimate",
}
FREE = r"\b(?:gratis|kostenlos\w*|free|umsonst)\b"
FREE_OFFERS = {"free_trial", "free_shipping", "free_quote"}
# Superlatives and health promises: legally risky in ads unless the business itself makes them.
RISKY = (r"führend\w*|marktführer|nummer (?:1|eins)|#\s?1|\bno\.? ?1\b|number one|\bleading\b|testsieger|"
         r"award[- ]winning|preisgekrönt|\bgesund\w*|\bhealth\w*|\bheil(?:t|en|ung)\b|\bcures?\b|medizin\w*|"
         r"abnehm\w*|weight loss|best price|bester preis|günstigste\w*|cheapest")


def _plain_numbers(text: str) -> str:
    """One notation for figures, so 1.727, 1,727 and 1727 compare equal, as do 327,27 and 327.27,
    and 900.0 and 900."""
    text = re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", text)   # thousands separators
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)               # decimal comma
    return re.sub(r"(?<=\d)\.0+(?!\d)", "", text)             # 900.0 is 900


def missing_numbers(text: str, source: str) -> list[str]:
    """Figures in `text` that do not appear in `source`.

    A number must appear in the source as that exact number ("20" inside "2026" does not count),
    and a percentage must appear there as a percentage.
    """
    low, src = _plain_numbers(text.lower()), _plain_numbers(source.lower())
    missing = []
    for number, percent in re.findall(r"(\d+(?:[.,]\d+)?)(\s?%)?", low):
        pattern = rf"(?<![\d.,]){re.escape(number)}(?![\d]|[.,]\d)" + (r"\s?%" if percent else "")
        if not re.search(pattern, src):
            missing.append(f"{number}{'%' if percent else ''}")
    return list(dict.fromkeys(missing))


def unsupported(text: str, offers: list[str], page_text: str) -> str:
    """Returns the reason a line is not allowed, or '' when the page supports it."""
    low, page = text.lower(), page_text.lower()
    missing = missing_numbers(text, page_text)
    if missing:
        return f"“{missing[0]}” is not on the page"
    for offer, pattern in OFFER_WORDS.items():
        match = re.search(pattern, low)
        if match and offer not in offers:
            return f"“{match.group(0)}” is not offered on the page"
    match = re.search(FREE, low)
    if match and not (FREE_OFFERS & set(offers)):
        return f"“{match.group(0)}” is not offered on the page"
    match = re.search(RISKY, low)
    if match and match.group(0) not in page:
        return f"“{match.group(0)}” is a claim the page does not make"
    return ""
