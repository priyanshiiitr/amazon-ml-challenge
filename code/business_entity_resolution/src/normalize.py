"""Text normalisation for business names and addresses.

Two properties drive the design.

*Cross-script.* Source 1 is 100% Latin; about 23% of Indian Source 2/3 names are
in one of nine Indic scripts. Everything is transliterated to Latin first (see
translit.py), because the previous `[^0-9a-z]` strip emptied those names
outright and made the records unretrievable.

*Country-agnostic.* The test set contains France, which never appears in
training, so no rule keys off the country label. The abbreviation tables are a
union of US / Indian / French conventions and an unseen country degrades to
plain accent-stripped cleaning rather than to nothing.

The `skel` key is the important one. It folds a token to a voicing- and
aspiration-insensitive consonant skeleton, which absorbs the three systematic
ways a transliterated name drifts from its English spelling:

    inherent-vowel insertion   devalapars / developers -> DVLPLS
    no voicing distinction     kulopal    / global     -> KLPL
    aspiration spelling        fud        / food       -> PT

Measured on 10,193 true (Latin S1, Indic S2/S3) ground-truth pairs, mean
token_set_ratio rises 4.3 -> 86.0 and 93.4% of pairs clear 60.
"""
from __future__ import annotations

import re
import unicodedata

from .translit import transliterate

# ---------------------------------------------------------------------------
# Legal suffixes / company-form tokens. Union of US, Indian and French forms
# plus common European ones, so an unseen country degrades gracefully.
# The transliterated Indic spellings are included: प्राइवेट लिमिटेड romanises
# to "praivet limited", and प्रा. लि. to "pra li".
# ---------------------------------------------------------------------------
LEGAL_TOKENS = {
    # anglophone
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "llc", "lc", "llp", "lp", "plc", "pllc", "pc", "ltda",
    "holdings", "holding", "group", "intl", "international",
    # indian, including romanised Indic forms
    "pvt", "private", "pl", "opc", "praivet", "praivate", "pra", "li",
    "limitet", "limitad", "limeted", "prailet", "elelpi", "prai",
    # french
    "sa", "sas", "sasu", "sarl", "sarlu", "eurl", "snc", "sci", "scop", "sca",
    "scs", "gie", "ei", "eirl", "etablissements", "ets",
    # german / dutch / nordic / iberian / italian, cheap insurance
    "gmbh", "mbh", "ag", "kg", "ohg", "ug", "bv", "nv", "as", "ab", "oy",
    "aps", "srl", "spa", "sl", "slu", "sau", "lda",
    # generic
    "the", "and", "of",
}

# Tokens that carry no discriminative signal in an address.
ADDR_STOPWORDS = {"the", "of", "and", "at", "in", "on", "a"}

# Address abbreviation expansion, applied token-wise. Where US and French
# conventions collide the more frequent one wins; these only ever feed fuzzy
# comparisons, so an occasional wrong expansion costs little.
ADDR_ABBREV = {
    # US street types
    "st": "street", "str": "street", "rd": "road", "ave": "avenue",
    "av": "avenue", "avn": "avenue", "blvd": "boulevard", "bvd": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "plz": "plaza",
    "sq": "square", "ter": "terrace", "trl": "trail", "pkwy": "parkway",
    "pky": "parkway", "hwy": "highway", "expy": "expressway", "cir": "circle",
    "cres": "crescent", "gdns": "gardens", "gdn": "garden", "mt": "mount",
    "ft": "fort", "jct": "junction", "xing": "crossing",
    # unit designators
    "apt": "apartment", "apts": "apartment", "ste": "suite", "fl": "floor",
    "flr": "floor", "flt": "flat", "bldg": "building", "blg": "building",
    "rm": "room", "dept": "department", "num": "number",
    "bsmt": "basement", "unit": "unit",
    # directionals
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    # indian
    "nr": "near", "opp": "opposite", "bh": "behind",
    "sec": "sector", "ph": "phase", "colny": "colony", "cly": "colony",
    "nag": "nagar", "ngr": "nagar", "mkt": "market", "extn": "extension",
    "ext": "extension", "soc": "society", "chs": "society",
    "mg": "mahatma gandhi", "vill": "village", "po": "post office",
    "distt": "district", "dist": "district", "tq": "taluka",
    # french
    "bd": "boulevard", "bld": "boulevard", "r": "rue", "imp": "impasse",
    "all": "allee", "chem": "chemin", "ch": "chemin", "rte": "route",
    "pl": "place", "res": "residence", "bat": "batiment",
    "esc": "escalier", "etg": "etage", "zi": "zone industrielle",
    "za": "zone artisanale", "bp": "boite postale",
}

# US state abbreviations -> full, so Source 3's "Texas" agrees with "TX".
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska",
    "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia",
}
# "in" and "or" collide with English words and "co" with the legal suffix, so
# they are deliberately absent above; the address context is not worth the risk.

_PUNCT_RE = re.compile(r"[^0-9a-z\s]+")
_WS_RE = re.compile(r"\s+")
_POSTAL_RE = re.compile(r"\b\d{5,6}\b")
_NUM_RE = re.compile(r"\b\d+[a-z]?\b")
_TLD_RE = re.compile(r"\b([a-z0-9-]+)\.(?:com|net|org|co|in|fr|biz|info|io)\b")
_RUN_RE = re.compile(r"(.)\1+")


def strip_accents(s: str) -> str:
    """cafe -> cafe, Learning -> Learning. Key for FR accents and IN noise."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def basic_clean(s) -> str:
    if not isinstance(s, str) or not s:
        return ""
    s = transliterate(s)              # Indic -> Latin, before anything strips it
    s = strip_accents(s.lower())
    s = s.replace("&", " and ").replace("+", " and ")
    s = s.replace("'", "").replace("’", "")   # o'brien -> obrien
    s = _TLD_RE.sub(r"\1", s)         # wilfordhancock.com -> wilfordhancock
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def norm_name(s) -> str:
    """Cleaned name with legal-form tokens removed (kept if that empties it)."""
    base = basic_clean(s)
    if not base:
        return ""
    kept = [t for t in base.split() if t not in LEGAL_TOKENS]
    return " ".join(kept) if kept else base


# ---------------------------------------------------------------------------
# Consonant skeleton
# ---------------------------------------------------------------------------
# Applied longest-first, before the single-character table.
_DIGRAPHS = (
    ("chh", "S"), ("sch", "S"), ("sh", "S"), ("ch", "S"),
    ("kh", "K"), ("gh", "K"), ("th", "T"), ("dh", "T"),
    ("ph", "P"), ("bh", "P"), ("ng", "N"), ("ny", "N"),
)
# Voicing and place folding: an Indic transliteration rarely preserves voicing,
# and Tamil does not encode it at all.
_SKEL_MAP = str.maketrans({
    "k": "K", "g": "K", "q": "K", "c": "K", "x": "K",
    "p": "P", "b": "P", "f": "P", "v": "P", "w": "P",
    "t": "T", "d": "T",
    "s": "S", "j": "S", "z": "S",
    "l": "L", "r": "L",
    "m": "M", "n": "M",
    "y": "", "h": "",
    "a": "", "e": "", "i": "", "o": "", "u": "",
})


def skel_token(t: str) -> str:
    for a, b in _DIGRAPHS:
        t = t.replace(a, b)
    t = t.translate(_SKEL_MAP)
    return _RUN_RE.sub(r"\1", t)      # doubled letters collapse


def skel(s: str) -> str:
    """Voicing-insensitive consonant skeleton, token-wise."""
    out = [skel_token(t) for t in (s or "").split()]
    return " ".join(t for t in out if t)


def norm_addr(s) -> str:
    """Cleaned address with abbreviations and US states expanded."""
    base = basic_clean(s)
    if not base:
        return ""
    out = []
    for t in base.split():
        if t in ADDR_ABBREV:
            out.append(ADDR_ABBREV[t])
        elif t in US_STATES:
            out.append(US_STATES[t])
        elif t in ADDR_STOPWORDS:
            continue
        else:
            out.append(t)
    return " ".join(out)


def addr_numbers(s: str) -> set[str]:
    """All standalone numeric tokens: street numbers, plot numbers, postal codes."""
    return set(_NUM_RE.findall(s or ""))


def addr_postal(s: str) -> str:
    """Longest 5-6 digit run: US ZIP / FR code postal (5) or Indian PIN (6)."""
    hits = _POSTAL_RE.findall(s or "")
    return max(hits, key=len) if hits else ""


def acronym(s: str) -> str:
    """'international business machines' -> 'ibm'. Catches IBM vs I.B.M."""
    toks = [t for t in (s or "").split() if t]
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


def sorted_key(s: str) -> str:
    """Order-invariant key, for word-order transpositions."""
    return " ".join(sorted((s or "").split()))


def normalise_frame(df, name_col="business_name", addr_col="business_address"):
    """Add every derived column the rest of the pipeline expects. In place."""
    df["n_name"] = df[name_col].map(norm_name)
    df["n_addr"] = df[addr_col].map(norm_addr)
    df["c_name"] = df[name_col].map(basic_clean)
    df["s_name"] = df["n_name"].map(skel)
    df["name_sorted"] = df["n_name"].map(sorted_key)
    df["name_acr"] = df["n_name"].map(acronym)
    df["postal"] = df["n_addr"].map(addr_postal)
    df["blob"] = (df["n_name"] + " " + df["n_addr"]).str.strip()
    return df
