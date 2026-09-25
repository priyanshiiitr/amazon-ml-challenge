"""Indic -> Latin transliteration, dependency-free.

Source 1 is 100% Latin. Roughly 23% of Indian Source 2/3 *names* are written in
a native script, spread over nine of them. The previous normaliser stripped
everything outside [0-9a-z], which mapped those names to the empty string and
made the records unretrievable -- about 9% of all S2/S3 rows.

Why this needs no data file or external package: the nine Indic blocks are all
ISCII-derived and laid out in parallel, each 0x80 apart --

    Devanagari 0900  Bengali 0980  Gurmukhi 0A00  Gujarati 0A80  Oriya 0B00
    Tamil      0B80  Telugu  0C00  Kannada  0C80  Malayalam 0D00

so a codepoint's offset within its block identifies the letter regardless of
script. We fold every block onto Devanagari, then romanise once.

The goal is *consistency*, not linguistic accuracy: both sides of a candidate
pair go through the same function, and what follows is fuzzy matching. Two
choices are deliberately lossy because they shorten the edit distance to the
English spelling the Source 1 record actually uses:

  * single vowels ("a" not "aa"), since these names are mostly English words
    respelled in an Indic script;
  * word-final schwa deletion, the standard Hindi rule.

    मार्केटिंग -> marketing      लिमिटेड -> limited      राम -> ram
"""
from __future__ import annotations

# Block starts, in the order the offsets align.
_BLOCKS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_DEVA = 0x0900

# --- Devanagari romanisation, indexed by offset from U+0900 ----------------
# Consonants are stored WITHOUT the inherent vowel; it is added by the driver.
_CONSONANT = {
    0x15: "k",  0x16: "kh", 0x17: "g",  0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh",0x1C: "j",  0x1D: "jh", 0x1E: "ny",
    0x1F: "t",  0x20: "th", 0x21: "d",  0x22: "dh", 0x23: "n",
    0x24: "t",  0x25: "th", 0x26: "d",  0x27: "dh", 0x28: "n",
    # 0x2B is romanised "f", not "ph": in these names it almost always carries
    # an English /f/ (फूड = food, फर्स्ट = first), and "f" is the shorter edit.
    0x29: "n",  0x2A: "p",  0x2B: "f",  0x2C: "b",  0x2D: "bh", 0x2E: "m",
    0x2F: "y",  0x30: "r",  0x31: "r",  0x32: "l",  0x33: "l",
    0x34: "l",  0x35: "v",  0x36: "sh", 0x37: "sh", 0x38: "s",  0x39: "h",
    # Sindhi/Marathi implosives and the Bengali/Assamese extras
    0x58: "k",  0x59: "kh", 0x5A: "g",  0x5B: "j",  0x5C: "r",  0x5D: "r",
    0x5E: "ph", 0x5F: "y",  0x79: "r",  0x7A: "y",  0x7B: "r",  0x7C: "r",
}

# Independent vowels.
_VOWEL = {
    0x05: "a",  0x06: "a",  0x07: "i",  0x08: "i",  0x09: "u",  0x0A: "u",
    0x0B: "ri", 0x0C: "li", 0x0D: "e",  0x0E: "e",  0x0F: "e",  0x10: "ai",
    0x11: "o",  0x12: "o",  0x13: "o",  0x14: "au", 0x60: "ri", 0x61: "li",
}

# Dependent vowel signs (matras) -- these REPLACE the inherent vowel.
_MATRA = {
    0x3E: "a",  0x3F: "i",  0x40: "i",  0x41: "u",  0x42: "u",
    0x43: "ri", 0x44: "ri", 0x45: "e",  0x46: "e",  0x47: "e",  0x48: "ai",
    0x49: "o",  0x4A: "o",  0x4B: "o",  0x4C: "au", 0x62: "ri", 0x63: "li",
}

_VIRAMA = 0x4D           # kills the inherent vowel
_NUKTA = 0x3C            # dot below; ignored
_AVAGRAHA = 0x3D         # ignored
_ANUSVARA = {0x01: "n", 0x02: "n", 0x03: "h", 0x00: "n"}  # candrabindu/anusvara/visarga
_DIGITS = {0x66 + i: str(i) for i in range(10)}

_INHERENT = "a"
_SKIP = {0x200B, 0x200C, 0x200D, 0xFEFF}   # ZWSP / ZWNJ / ZWJ / BOM


def _fold(cp: int) -> int:
    """Map any Indic codepoint onto its Devanagari counterpart."""
    for base in _BLOCKS:
        if base <= cp < base + 0x80:
            return _DEVA + (cp - base)
    return cp


def has_indic(s: str) -> bool:
    for ch in s:
        if 0x0900 <= ord(ch) < 0x0D80:
            return True
    return False


def transliterate(s: str) -> str:
    """Romanise any Indic runs in `s`, leaving other characters untouched."""
    if not s or not has_indic(s):
        return s

    out: list[str] = []
    pending = False          # a consonant is awaiting its inherent vowel

    def flush():
        nonlocal pending
        if pending:
            out.append(_INHERENT)
            pending = False

    def drop_final_schwa():
        """Hindi deletes the word-final inherent vowel: राम is 'ram', not 'rama'."""
        nonlocal pending
        pending = False

    for ch in s:
        cp = ord(ch)
        if cp in _SKIP:
            continue
        folded = _fold(cp)
        if not (_DEVA <= folded < _DEVA + 0x80):
            # non-Indic: ends the current syllable
            if ch.isalnum():
                flush()
            else:
                drop_final_schwa()
            out.append(ch)
            continue

        off = folded - _DEVA
        if off in _CONSONANT:
            flush()
            out.append(_CONSONANT[off])
            pending = True
        elif off in _MATRA:
            pending = False           # the matra supplies the vowel instead
            out.append(_MATRA[off])
        elif off == _VIRAMA:
            pending = False           # explicit vowel suppression
        elif off in _VOWEL:
            flush()
            out.append(_VOWEL[off])
        elif off in _ANUSVARA:
            flush()
            out.append(_ANUSVARA[off])
        elif off in _DIGITS:
            flush()
            out.append(_DIGITS[off])
        elif off in (_NUKTA, _AVAGRAHA):
            continue
        else:
            flush()                   # unknown sign: drop it, keep the syllable

    drop_final_schwa()
    return "".join(out)
