import os
import re
import urllib.parse
from difflib import SequenceMatcher
from datetime import datetime

import requests
import feedparser
from bs4 import BeautifulSoup
from flask import Flask, render_template, request

# ---- Optional ML (auto-skip if files missing) ----
import pickle

app = Flask(__name__)

# ==========================================
# CONFIG
# ==========================================
# Put your real GNews API key here or set env var GNEWS_API_KEY
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "put your API here")

# Where your existing ML files live (optional)
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "model.pkl")
ML_VECT_PATH = os.getenv("ML_VECT_PATH", "vectorizer.pkl")

# Credible outlets (add/remove as you like)
CREDIBLE_DOMAINS = {
    "bbc.com": "BBC",
    "reuters.com": "Reuters",
    "apnews.com": "AP News",
    "theguardian.com": "The Guardian",
    "nytimes.com": "NYTimes",
    "wsj.com": "WSJ",
    "cnn.com": "CNN",
    "aljazeera.com": "Al Jazeera",
    "thehindu.com": "The Hindu",
    "indianexpress.com": "Indian Express",
    "hindustantimes.com": "Hindustan Times",
    "ndtv.com": "NDTV",
    "timesofindia.indiatimes.com": "Times of India",
    "indiatoday.in": "India Today",
    "economictimes.indiatimes.com": "Economic Times",
    "livehindustan.com": "Live Hindustan",
    "business-standard.com": "Business Standard",
}

# Fact-checking outlets
FACT_CHECK_DOMAINS = {
    "altnews.in": "Alt News",
    "boomlive.in": "BOOM",
    "factcheck.org": "FactCheck.org",
    "politifact.com": "PolitiFact",
    "snopes.com": "Snopes",
    "aap.com.au": "AAP FactCheck",
    "afp.com": "AFP Fact Check",
}

# Words that suggest debunks in fact-check headlines
FACT_NEGATIVE_HINTS = [
    "fake", "false", "fabricated", "misleading", "debunked", "not true",
    "hoax", "incorrect", "bogus", "no, ", "fact check", "fact-check", "factcheck"
]

# Simple English stopwords for keywording (kept small intentionally)
STOPWORDS = set("""
a an the of to in on for and or with from that this those these is are was were be been being by at as it its into about
""".split())


# ==========================================
# OPTIONAL ML LOADING (graceful if missing)
# ==========================================
ml_model = None
ml_vect = None
try:
    if os.path.exists(ML_MODEL_PATH) and os.path.exists(ML_VECT_PATH):
        with open(ML_MODEL_PATH, "rb") as f:
            ml_model = pickle.load(f)
        with open(ML_VECT_PATH, "rb") as f:
            ml_vect = pickle.load(f)
        print("[ML] Loaded model/vectorizer.")
    else:
        print("[ML] Model/vectorizer not found. Skipping ML fallback.")
except Exception as e:
    print("[ML] Failed to load model/vectorizer:", e)
    ml_model, ml_vect = None, None


# ==========================================
# UTILS
# ==========================================
def clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def extract_title_from_url(url: str, timeout=7) -> str:
    """Fetch <title> from a URL (safer than heavy libs)."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string if soup.title else ""
        return clean_text(title)
    except Exception:
        # fallback to URL slug
        try:
            slug = url.split("/")[-1]
            slug = slug.replace("-", " ").replace("_", " ")
            return clean_text(slug)
        except Exception:
            return ""

def keywordize(text: str, max_words=10) -> str:
    """Take the most meaningful words (very simple) to form a search query."""
    words = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    kept = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if not kept:
        kept = words
    return " ".join(kept[:max_words]) if kept else text

def similarity(a: str, b: str) -> float:
    a = (a or "").lower()
    b = (b or "").lower()
    return SequenceMatcher(None, a, b).ratio()

def domain_from_url(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


# ==========================================
# SEARCHERS
# ==========================================
def search_gnews(query: str, lang="en", max_results=10, timeout=8):
    """
    Uses GNews official API (gnews.io). Requires token=GNEWS_API_KEY.
    """
    if not GNEWS_API_KEY or GNEWS_API_KEY == "YOUR_GNEWS_API_KEY":
        return []

    try:
        # Let requests handle encoding via params
        params = {
            "q": query,
            "lang": lang,
            "max": str(max_results),
            "token": GNEWS_API_KEY,
        }
        r = requests.get("https://gnews.io/api/v4/search", params=params, timeout=timeout)
        data = r.json() if r.status_code == 200 else {}
        out = []
        for art in data.get("articles", []):
            out.append({
                "title": clean_text(art.get("title") or ""),
                "url": art.get("url") or "",
                "source": (art.get("source") or {}).get("name") or domain_from_url(art.get("url") or ""),
            })
        return out
    except Exception as e:
        print("[GNews] error:", e)
        return []

def search_rss(query: str, max_items=12):
    """
    Google News RSS (no key).
    """
    try:
        encoded = urllib.parse.quote_plus(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded}"
        feed = feedparser.parse(rss_url)
        out = []
        for entry in feed.entries[:max_items]:
            out.append({
                "title": clean_text(getattr(entry, "title", "")),
                "url": getattr(entry, "link", ""),
                "source": getattr(entry, "source", {}).get("title", "Unknown") if hasattr(entry, "source") else "Unknown",
            })
        return out
    except Exception as e:
        print("[RSS] error:", e)
        return []


# ==========================================
# CORE VERIFICATION LOGIC
# ==========================================
def verify_news(user_input: str):
    """
    Returns: dict with keys:
      verdict: "REAL" | "FAKE" | "UNVERIFIED"
      confidence: 0-100
      sources: list[{title, url, source}]
      fact_sources: list[...]  (if any)
      used: which signals were used
    """
    used_signals = []
    raw_input = clean_text(user_input)

    # 1) Normalize input → query string
    if raw_input.startswith("http://") or raw_input.startswith("https://"):
        headline = extract_title_from_url(raw_input)
        query = keywordize(headline if headline else raw_input)
        used_signals.append("url->title")
    else:
        headline = raw_input
        query = keywordize(raw_input)
        used_signals.append("text/headline")

    # 2) Hit GNews first, fallback to RSS
    gnews_results = search_gnews(query)
    if gnews_results:
        used_signals.append("gnews")
    rss_results = search_rss(query)
    if rss_results:
        used_signals.append("rss")

    combined = (gnews_results or []) + (rss_results or [])
    # Deduplicate by URL
    seen = set()
    deduped = []
    for item in combined:
        u = item["url"]
        if u and u not in seen:
            seen.add(u)
            deduped.append(item)

    # 3) Split into credible + fact-check matches with similarity filtering
    credible_hits = []
    fact_hits = []

    # similarity thresholds (headline vs returned title)
    # if input was URL, we likely have a decent headline → higher threshold
    sim_thresh = 0.55 if headline else 0.45

    for it in deduped:
        url = it["url"]
        title = it["title"]
        dom = domain_from_url(url)
        sim = similarity(headline or query, title)

        # Fact-checkers first
        if any(fc in dom for fc in FACT_CHECK_DOMAINS):
            # If their title suggests debunk, treat as FAKE proof
            lower_title = title.lower()
            if any(hint in lower_title for hint in FACT_NEGATIVE_HINTS):
                fact_hits.append(it)
            else:
                # Keep neutral fact-check links too (could confirm REAL or context)
                fact_hits.append(it)
            continue

        # Credible media (with headline similarity)
        if any(cd in dom for cd in CREDIBLE_DOMAINS) and sim >= sim_thresh:
            credible_hits.append(it)

    # 4) Decision rules
    # FAKE if a reputable fact-checker explicitly debunks
    debunk_hits = []
    for fh in fact_hits:
        if any(h in fh["title"].lower() for h in FACT_NEGATIVE_HINTS):
            debunk_hits.append(fh)

    if debunk_hits:
        # Confidence based on number of debunks
        conf = min(98, 80 + len(debunk_hits) * 6)
        return {
            "verdict": "FAKE",
            "confidence": conf,
            "sources": credible_hits,   # credible coverage (if any)
            "fact_sources": debunk_hits,
            "used": used_signals
        }

    # REAL if multiple credible outlets report similar headline
    if len(credible_hits) >= 2:
        conf = min(96, 70 + len(credible_hits) * 6)
        return {
            "verdict": "REAL",
            "confidence": conf,
            "sources": credible_hits,
            "fact_sources": fact_hits,
            "used": used_signals
        }

    # LIKELY REAL if exactly 1 credible match (but not enough consensus)
    if len(credible_hits) == 1:
        conf = 60
        return {
            "verdict": "REAL",
            "confidence": conf,
            "sources": credible_hits,
            "fact_sources": fact_hits,
            "used": used_signals
        }

    # 5) Fallback to ML if available and APIs can’t verify
    if ml_model and ml_vect:
        try:
            text_for_ml = headline if headline else raw_input
            X = ml_vect.transform([text_for_ml])
            proba = None
            if hasattr(ml_model, "predict_proba"):
                proba = ml_model.predict_proba(X)[0]
                # Assume proba order [FAKE, REAL] or [0,1]—adjust if needed
                if len(proba) == 2:
                    fake_score = float(proba[0])
                    real_score = float(proba[1])
                else:
                    # If it's a single-prob output, assume it's "REAL"
                    fake_score, real_score = (1 - float(proba[0])), float(proba[0])
            else:
                # No proba method; use decision_function or predict
                pred = ml_model.predict(X)[0]
                fake_score = 0.5
                real_score = 0.5
                if str(pred).lower() in ["real", "true", "1"]:
                    real_score = 0.8
                else:
                    fake_score = 0.8

            if real_score >= fake_score:
                verdict = "REAL"
                conf = int(round(real_score * 100))
            else:
                verdict = "FAKE"
                conf = int(round(fake_score * 100))

            used_signals.append("ml")
            return {
                "verdict": verdict,
                "confidence": conf,
                "sources": credible_hits,  # likely empty
                "fact_sources": fact_hits, # likely empty
                "used": used_signals
            }
        except Exception as e:
            print("[ML] prediction error:", e)

    # 6) Nothing found → UNVERIFIED
    return {
        "verdict": "UNVERIFIED",
        "confidence": 0,
        "sources": [],
        "fact_sources": fact_hits,  # we keep neutral fact links if any
        "used": used_signals
    }


# ==========================================
# FLASK ROUTES
# ==========================================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_input = request.form.get("news", "").strip()
        if user_input:
            result = verify_news(user_input)
            result["input"] = user_input
            result["checked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template("index.html", result=result)


if __name__ == "__main__":
    # Run with:  set GNEWS_API_KEY=...  (Windows)
    #            export GNEWS_API_KEY=... (Mac/Linux)
    app.run(debug=True)
