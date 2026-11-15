"""Import real search volumes and bids from a Google Keyword Planner export (or any simple CSV).

Keyword Planner is free with a Google Ads account: Tools > Keyword Planner > Get search volume,
paste the keyword list this tool produces, then download as CSV.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

ALIASES = {
    "keyword": ["keyword", "search term", "suchbegriff"],
    "volume": ["avg. monthly searches", "average monthly searches", "search volume", "volume",
               "durchschn. suchanfragen pro monat", "suchvolumen"],
    "cpc_low": ["top of page bid (low range)", "gebot für obere positionen (niedriger bereich)", "cpc low"],
    "cpc_high": ["top of page bid (high range)", "gebot für obere positionen (hoher bereich)", "cpc high", "cpc"],
    "competition": ["competition", "wettbewerb"],
}


def _number(value: str) -> float | None:
    """Handles '1,300', '1.300', '1,30', '0.85', '1K – 10K' (takes the lower bound of a range)."""
    value = (value or "").strip()
    if not value or value in ("-", "–"):
        return None
    value = re.split(r"\s*[–-]\s*", value)[0] if re.search(r"\d\s*[–-]\s*\d", value) else value
    mult = 1.0
    match = re.search(r"([\d.,]+)\s*([KkMm])?", value)
    if not match:
        return None
    digits, suffix = match.group(1), match.group(2)
    if suffix:
        mult = 1000.0 if suffix.lower() == "k" else 1_000_000.0
    if "," in digits and "." in digits:
        digits = digits.replace(".", "").replace(",", ".") if digits.rfind(",") > digits.rfind(".") else digits.replace(",", "")
    elif "," in digits:
        head, tail = digits.rsplit(",", 1)
        digits = head.replace(",", "") + ("." + tail if len(tail) != 3 else tail)
    elif "." in digits:
        head, tail = digits.rsplit(".", 1)
        if len(tail) == 3 and head.isdigit():
            digits = head + tail  # thousands separator, as in 1.300
    try:
        return float(digits) * mult
    except ValueError:
        return None


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_keyword_csv(source: str | Path | bytes) -> dict[str, dict]:
    """Returns {keyword: {volume, cpc_low, cpc_high, competition}} with missing values left out."""
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    lines = _decode(raw).splitlines()
    header_at = next((i for i, line in enumerate(lines)
                      if any(a in line.lower() for a in ALIASES["keyword"]) and ("\t" in line or "," in line or ";" in line)),
                     None)
    if header_at is None:
        raise ValueError("No header row with a 'Keyword' column was found in the CSV.")
    header = lines[header_at]
    delimiter = max(("\t", ";", ","), key=header.count)
    reader = csv.reader(io.StringIO("\n".join(lines[header_at:])), delimiter=delimiter)
    columns = [c.strip().lower() for c in next(reader)]

    def find(field: str) -> int | None:
        for alias in ALIASES[field]:
            if alias in columns:
                return columns.index(alias)
        return None

    index = {name: find(name) for name in ALIASES}
    if index["keyword"] is None:
        raise ValueError("The CSV has no 'Keyword' column.")
    out: dict[str, dict] = {}
    for row in reader:
        if len(row) <= index["keyword"]:
            continue
        keyword = row[index["keyword"]].strip().lower()
        if not keyword:
            continue
        entry: dict = {}
        for name in ("volume", "cpc_low", "cpc_high"):
            col = index[name]
            number = _number(row[col]) if col is not None and col < len(row) else None
            if number is not None:
                entry[name] = number
        col = index["competition"]
        if col is not None and col < len(row) and row[col].strip():
            entry["competition"] = row[col].strip()
        out[keyword] = entry
    return out
