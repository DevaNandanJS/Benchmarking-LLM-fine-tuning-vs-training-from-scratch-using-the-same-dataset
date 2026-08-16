"""Phase 1 — Data Extraction & Cleaning."""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Insert repo root on path so `scripts.*` imports work when run directly
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from TASK1_finetuning_model.scripts.common import set_seed  # noqa: E402
from TASK1_finetuning_model.scripts.config import (  # noqa: E402
    CLEAN_TXT,
    EXTRACTED_DIR,
    HYPHEN_LOG,
    MANIFEST_JSON,
    RAW_PAGES_DIR,
    RAW_PDF,
    STATS_JSON,
)

set_seed()  # determinism convention (§0)

# Garble thresholds (tunable)
# Either signal tripping marks a page as potentially garbled.
_GARBLE_NON_ASCII_THRESHOLD = 0.30   # > 30 % non-ASCII chars  -> suspect mojibake
_GARBLE_NON_DICT_THRESHOLD = 0.40   # > 40 % non-dictionary English words -> suspect font-remap

# Header/footer repetition threshold
# A line must appear on at least this fraction of pages to be considered a
# running header/footer (not a section heading).
_REPEAT_THRESHOLD = 0.40

# Wordlist (for garble check + hyphen disambiguation)

def _load_wordlist() -> set[str]:
    """Load an English word set from nltk.corpus.words. Gracefully degrades."""
    try:
        import nltk  # type: ignore
        try:
            from nltk.corpus import words as _w
            word_set = set(w.lower() for w in _w.words())
        except LookupError:
            print("[Phase 1] Downloading nltk 'words' corpus…")
            nltk.download("words", quiet=True)
            from nltk.corpus import words as _w
            word_set = set(w.lower() for w in _w.words())
        print(f"[Phase 1] Loaded wordlist ({len(word_set):,} entries).")
        return word_set
    except ImportError:
        print("[Phase 1] WARNING: nltk not installed. Wordlist-based checks disabled.")
        return set()

WORDLIST: set[str] = _load_wordlist()

# Extraction

def _extract_pdfplumber(pdf_path: Path) -> list[str]:
    """Extract page texts via pdfplumber."""
    import pdfplumber  # type: ignore
    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages

def _extract_pypdf(pdf_path: Path) -> list[str]:
    """Extract page texts via pypdf."""
    from pypdf import PdfReader  # type: ignore
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]

# Quality / garble scoring

def _quality_scores(text: str) -> dict:
    """Return two independent garbling signals for *text*:."""
    if not text.strip():
        return {"non_ascii_ratio": 0.0, "non_dict_word_ratio": 0.0, "char_count": 0}

    chars = text
    non_ascii_ratio = sum(1 for c in chars if ord(c) > 127) / max(len(chars), 1)

    alpha_tokens = re.findall(r"[a-zA-Z]{2,}", text)  # length ≥ 2 avoids noise
    if alpha_tokens and WORDLIST:
        non_dict_ratio = sum(
            1 for w in alpha_tokens if w.lower() not in WORDLIST
        ) / len(alpha_tokens)
    else:
        non_dict_ratio = 0.0  # can't assess without wordlist

    return {
        "non_ascii_ratio": round(non_ascii_ratio, 4),
        "non_dict_word_ratio": round(non_dict_ratio, 4),
        "char_count": len(text),
    }

def _is_garbled(scores: dict) -> bool:
    return (
        scores["non_ascii_ratio"] > _GARBLE_NON_ASCII_THRESHOLD
        or scores["non_dict_word_ratio"] > _GARBLE_NON_DICT_THRESHOLD
    )

# Header / footer detection

# Page-number patterns — safe to strip unconditionally via regex:
#   "3", "  42  ", "Page 3", "Page 3 of 10"
_PAGE_NUM_RE = re.compile(
    r"^\s*(\d+|[Pp]age\s+\d+(\s+of\s+\d+)?)\s*$"
)

def _build_repeating_line_set(all_pages: list[str]) -> set[str]:
    """Identify running header/footer strings by frequency across pages."""
    total = len(all_pages)
    if total == 0:
        return set()

    exact_page_hits: Counter = Counter()   # line  -> number of pages it appears on
    norm_page_hits: Counter = Counter()    # normalised -> number of pages
    norm_to_example: dict[str, str] = {}  # normalised -> one representative original

    for page_text in all_pages:
        seen_on_page: set[str] = set()
        seen_norm_on_page: set[str] = set()
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line not in seen_on_page:
                exact_page_hits[line] += 1
                seen_on_page.add(line)
            norm = re.sub(r"\d+", "N", line)
            if norm not in seen_norm_on_page:
                norm_page_hits[norm] += 1
                norm_to_example.setdefault(norm, line)
                seen_norm_on_page.add(norm)

    repeating: set[str] = set()
    for line, count in exact_page_hits.items():
        if count / total >= _REPEAT_THRESHOLD:
            repeating.add(line)
    # Add the representative example for each frequent normalised pattern
    for norm, count in norm_page_hits.items():
        if count / total >= _REPEAT_THRESHOLD:
            repeating.add(norm_to_example[norm])

    return repeating

# Cleaning

def _clean_page(
    text: str,
    repeating_lines: set[str],
    join_log_lines: list[str],
) -> str:
    """Clean a single page's raw text."""
    # 1 & 2: drop page-number and repeating header/footer lines
    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            kept_lines.append(raw_line)  # keep blank lines (collapse later)
            continue
        # Page-number lines: safe unconditional regex drop
        if _PAGE_NUM_RE.match(stripped):
            continue
        # Exact header/footer match
        if stripped in repeating_lines:
            continue
        # Digit-normalised header/footer match
        if re.sub(r"\d+", "N", stripped) in {re.sub(r"\d+", "N", r) for r in repeating_lines}:
            continue
        kept_lines.append(raw_line)
    text = "\n".join(kept_lines)

    # 3: Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 4: Soft-hyphen (U+00AD) line-end joins — unambiguous, always drop hyphen
    # U+00AD is explicitly a discretionary hyphen — its only purpose is layout;
    # joining without the character is always semantically correct.
    text = re.sub(r"\xad[ \t]*\n[ \t]*", "", text)

    # 5: ASCII hyphen line-end joins — structural heuristic

    def _maybe_join_hyphen(m: re.Match) -> str:
        pre = m.group(1)    # word-part before the hyphen
        post = m.group(2)   # word-part after the line break
        joined = pre + post
        hyphenated = pre + "-" + post

        # Case 1: digit-letter or letter-digit at the boundary -> keep hyphen
        # (model names like GPT-4o, CodeLlama-13B, ROUGE-2)
        if pre[-1].isdigit() or post[0].isdigit():
            decision = f"KEEP_HYPHEN  : {m.group(0)!r:35s}  ->  {hyphenated!r}  (heuristic: digit boundary)"
            result = hyphenated
        # Case 2: short syllable on left (≤3 chars) -> layout break, drop hyphen
        elif len(pre) <= 3:
            decision = f"DROP_HYPHEN  : {m.group(0)!r:35s}  ->  {joined!r}  (heuristic: short-pre syllable, len={len(pre)})"
            result = joined
        # Case 3: short syllable on right (≤3 chars) -> layout break, drop hyphen
        elif len(post) <= 3:
            decision = f"DROP_HYPHEN  : {m.group(0)!r:35s}  ->  {joined!r}  (heuristic: short-post syllable, len={len(post)})"
            result = joined
        # Case 4: both sides are ≥4 chars -> likely a real compound, keep hyphen
        else:
            decision = f"KEEP_HYPHEN  : {m.group(0)!r:35s}  ->  {hyphenated!r}  (heuristic: both-sides ≥4 chars)"
            result = hyphenated

        join_log_lines.append(decision)
        return result

    text = re.sub(r"(\w+)-[ \t]*\n[ \t]*(\w+)", _maybe_join_hyphen, text)

    return text

def _collapse_blank_lines(text: str, max_consecutive: int = 2) -> str:
    """Collapse runs of more than *max_consecutive* blank lines to exactly that many."""
    # A "blank line" is a line containing only whitespace.
    # Replace runs of (max_consecutive + 1) or more blank lines.
    pattern = r"([ \t]*\n){%d,}" % (max_consecutive + 2)
    replacement = "\n" * (max_consecutive + 1)
    return re.sub(pattern, replacement, text)

# Main

def main() -> None:
    print(f"[Phase 1] Source PDF : {RAW_PDF}")
    if not RAW_PDF.exists():
        print(f"[ERROR] PDF not found: {RAW_PDF}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directories exist
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Extract with pdfplumber
    print("[Phase 1] Extracting with pdfplumber…")
    try:
        plumber_pages = _extract_pdfplumber(RAW_PDF)
        print(f"[Phase 1]   pdfplumber: {len(plumber_pages)} pages")
    except Exception as exc:
        print(f"[Phase 1] WARNING: pdfplumber failed ({exc}); will use pypdf only.")
        plumber_pages = []

    # Extract with pypdf
    print("[Phase 1] Extracting with pypdf…")
    try:
        pypdf_pages = _extract_pypdf(RAW_PDF)
        print(f"[Phase 1]   pypdf     : {len(pypdf_pages)} pages")
    except Exception as exc:
        print(f"[Phase 1] WARNING: pypdf failed ({exc}).")
        pypdf_pages = []

    if not plumber_pages and not pypdf_pages:
        print("[ERROR] Both extractors failed. Cannot continue.", file=sys.stderr)
        sys.exit(1)

    # Align to same page count (pad shorter list with empty strings)
    n_pages = max(len(plumber_pages), len(pypdf_pages))
    plumber_pages += [""] * (n_pages - len(plumber_pages))
    pypdf_pages += [""] * (n_pages - len(pypdf_pages))

    # Save raw per-page output for audit trail
    print(f"[Phase 1] Saving raw page outputs to {RAW_PAGES_DIR} …")
    for i, (pb, pp) in enumerate(zip(plumber_pages, pypdf_pages)):
        (RAW_PAGES_DIR / f"page_{i+1:04d}_pdfplumber.txt").write_text(pb, encoding="utf-8")
        (RAW_PAGES_DIR / f"page_{i+1:04d}_pypdf.txt").write_text(pp, encoding="utf-8")

    # Per-page quality scoring and extractor selection
    print("[Phase 1] Scoring page quality and selecting extractor per page…")
    manifest: list[dict] = []
    selected_pages: list[str] = []

    for i, (pb, pp) in enumerate(zip(plumber_pages, pypdf_pages)):
        pb_scores = _quality_scores(pb)
        pp_scores = _quality_scores(pp)
        pb_garbled = _is_garbled(pb_scores)
        pp_garbled = _is_garbled(pp_scores)

        # Combined score: lower = cleaner (used as tiebreaker)
        pb_combined = pb_scores["non_ascii_ratio"] + pb_scores["non_dict_word_ratio"]
        pp_combined = pp_scores["non_ascii_ratio"] + pp_scores["non_dict_word_ratio"]

        if not pb_garbled:
            chosen, text, reason = "pdfplumber", pb, "primary_clean"
        elif not pp_garbled:
            chosen, text, reason = "pypdf", pp, "fallback_pdfplumber_garbled"
            print(
                f"[Phase 1]   Page {i+1:3d}: pdfplumber garbled "
                f"(non_ascii={pb_scores['non_ascii_ratio']:.1%}, "
                f"non_dict={pb_scores['non_dict_word_ratio']:.1%}); using pypdf."
            )
        else:
            # Both garbled — pick whichever is less bad; flag for manual review
            if pb_combined <= pp_combined:
                chosen, text, reason = "pdfplumber", pb, "both_garbled_pdfplumber_less_bad"
            else:
                chosen, text, reason = "pypdf", pp, "both_garbled_pypdf_less_bad"
            print(
                f"[Phase 1]   Page {i+1:3d}: WARNING — BOTH extractors garbled. "
                f"Manual spot-check recommended."
            )

        final_scores = _quality_scores(text)
        selected_pages.append(text)
        manifest.append(
            {
                "page": i + 1,
                "extractor_chosen": chosen,
                "reason": reason,
                "garbled_flag": _is_garbled(final_scores),
                "final_scores": final_scores,
                "pdfplumber_scores": pb_scores,
                "pypdf_scores": pp_scores,
            }
        )

    # Detect repeating header/footer lines
    print("[Phase 1] Detecting repeating header/footer lines…")
    repeating_lines = _build_repeating_line_set(selected_pages)
    if repeating_lines:
        print(
            f"[Phase 1]   Found {len(repeating_lines)} repeating pattern(s) to strip "
            f"(threshold: {_REPEAT_THRESHOLD:.0%} of pages):"
        )
        for r in sorted(repeating_lines)[:15]:  # show first 15
            print(f"    {r!r}")
    else:
        print("[Phase 1]   No repeating header/footer patterns detected.")

    # Clean each selected page
    print("[Phase 1] Cleaning pages…")
    join_log_lines: list[str] = []
    cleaned_pages = [
        _clean_page(page_text, repeating_lines, join_log_lines)
        for page_text in selected_pages
    ]

    # Concatenate and post-process
    full_text = "\n\n".join(p for p in cleaned_pages if p.strip())
    full_text = _collapse_blank_lines(full_text, max_consecutive=2)
    full_text = full_text.strip()

    # Write outputs
    CLEAN_TXT.write_text(full_text, encoding="utf-8")
    print(f"[Phase 1] Wrote {CLEAN_TXT.relative_to(CLEAN_TXT.parents[3])}  "
          f"({len(full_text):,} chars)")

    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[Phase 1] Wrote {MANIFEST_JSON.name}  ({n_pages} pages logged)")

    if join_log_lines:
        HYPHEN_LOG.write_text("\n".join(join_log_lines) + "\n", encoding="utf-8")
        print(f"[Phase 1] Wrote {HYPHEN_LOG.name}  ({len(join_log_lines)} hyphen-join decisions)")
    else:
        print("[Phase 1]   No hyphen-join decisions to log.")

    # Compute stats
    char_count = len(full_text)
    word_count = len(full_text.split())

    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("gpt2")
        proxy_token_count: int | None = len(enc.encode(full_text))
        print(f"[Phase 1] Proxy token count (GPT-2/tiktoken): {proxy_token_count:,}")
    except ImportError:
        proxy_token_count = None
        print("[Phase 1] WARNING: tiktoken not installed; proxy_token_count_gpt2_tiktoken = null.")

    n_garbled = sum(1 for m in manifest if m["garbled_flag"])
    stats = {
        "char_count": char_count,
        "word_count": word_count,
        # Explicitly labelled to avoid confusion with Phase 2's real model tokenizer count
        "proxy_token_count_gpt2_tiktoken": proxy_token_count,
        "note": (
            "proxy_token_count_gpt2_tiktoken uses the GPT-2 tiktoken encoding as a "
            "lightweight stand-in. Phase 2 will recompute and overwrite this field "
            "using the actual base-model tokenizer (may differ by ±5-15 %)."
        ),
        "pages_total": n_pages,
        "pages_garbled_flag": n_garbled,
        "repeating_patterns_stripped": len(repeating_lines),
        "hyphen_joins_logged": len(join_log_lines),
        "repeat_threshold_used": _REPEAT_THRESHOLD,
        "garble_thresholds": {
            "non_ascii": _GARBLE_NON_ASCII_THRESHOLD,
            "non_dict_word": _GARBLE_NON_DICT_THRESHOLD,
        },
    }
    STATS_JSON.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[Phase 1] Wrote {STATS_JSON.name}")
    print(
        f"\n[Phase 1] Summary:\n"
        f"  chars          : {char_count:,}\n"
        f"  words          : {word_count:,}\n"
        f"  proxy tokens   : {proxy_token_count if proxy_token_count is not None else 'N/A'}\n"
        f"  pages total    : {n_pages}\n"
        f"  pages flagged  : {n_garbled}\n"
        f"  header/footer patterns stripped: {len(repeating_lines)}\n"
        f"  hyphen joins logged: {len(join_log_lines)}\n"
    )
    if n_garbled:
        print(
            f"[Phase 1] ACTION REQUIRED: {n_garbled} page(s) flagged as potentially garbled.\n"
            f"  Check {MANIFEST_JSON.name} for page numbers, then spot-check the\n"
            f"  corresponding files in raw_pages/ and the final document_clean.txt."
        )
    print(
        "[Phase 1] Done. Please open data/extracted/document_clean.txt\n"
        "          and spot-check a few random sections before proceeding to Phase 2."
    )

if __name__ == "__main__":
    main()
