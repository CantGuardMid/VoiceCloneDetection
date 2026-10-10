"""
text_authenticity.py
---------------------
Heuristic "does this read like an AI-written script?" checker for a call
transcript -- the same idea as QuillBot's / GPTZero's AI-content detectors,
applied to what a caller said.

WHY THIS EXISTS: voice-cloning scams increasingly script the call with an
LLM first ("sound urgent, mention a locked account, ask for a code") and
either read it or have a cloned voice speak it. A transcript that reads
like polished AI copy -- uniform sentence lengths, stock transitional
phrases, no natural disfluencies -- is itself a signal worth surfacing next
to the audio-based checks.

HONESTY NOTE: real detectors (GPTZero, QuillBot, Originality.ai) score
text by running it through a language model and measuring *perplexity* and
*burstiness* (how surprised the LM is by the text, and how much that
surprise varies sentence to sentence). That requires a trained LM; this
sandbox has no internet access to download one (GPT-2 weights, etc.), so
this module uses transparent, rule-based stylistic heuristics instead --
sentence-length uniformity, stock AI/marketing-copy phrases, lexical
diversity, and presence/absence of natural speech disfluencies ("um",
"like", false starts). It is a reasonable proxy for the same signal, not
a trained perplexity model. Swap in a real LM-based scorer (e.g. a small
GPT-2 + perplexity calculation) for production.
"""

import re

# Common stock phrases that show up disproportionately often in
# LLM-generated text/marketing copy. This list is intentionally small and
# generic (transition words / hedging phrases), not tied to any one model.
AI_STOCK_PHRASES = [
    "furthermore", "moreover", "additionally", "in conclusion", "in summary",
    "it is important to note", "it's important to note", "on the other hand",
    "overall,", "ultimately,", "in today's world", "in this day and age",
    "let's delve into", "delve into", "i hope this helps", "certainly!",
    "here are some", "as an ai", "as a language model", "it's worth noting",
    "please note that", "rest assured", "at your earliest convenience",
    "we kindly request", "for security purposes", "to ensure the safety of",
    "failure to comply", "immediate action is required", "verify your identity",
]

# Natural-speech disfluencies / hedges that show up in spontaneous human
# speech but rarely in scripted/AI-generated text.
DISFLUENCY_MARKERS = [
    r"\bum\b", r"\buh\b", r"\ber+\b", r"\bhmm+\b", r"\byou know\b",
    r"\bi mean\b", r"\blike,?\b", r"\bkind of\b", r"\bsort of\b",
    r"\.\.\.", r"--", r"\bwell,\b",
]


def _sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _words(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())


def analyze_text(text: str) -> dict:
    text = text.strip()
    if not text:
        return {"error": "empty text"}

    sentences = _sentences(text)
    words = _words(text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)

    if word_count < 15:
        return {
            "error": "too_short",
            "message": "Need at least ~15 words for a meaningful style read.",
            "word_count": word_count,
        }

    # --- 1. sentence-length uniformity (burstiness) ---
    sent_lens = [len(_words(s)) for s in sentences] or [word_count]
    mean_len = sum(sent_lens) / len(sent_lens)
    variance = sum((l - mean_len) ** 2 for l in sent_lens) / len(sent_lens)
    std_len = variance ** 0.5
    burstiness = std_len / (mean_len + 1e-6)
    # low burstiness (uniform sentence lengths) -> more AI-like
    uniformity_score = max(0.0, min(1.0, 1.0 - burstiness))  # 0=varied(human-like), 1=uniform(AI-like)

    # --- 2. lexical diversity (type-token ratio) ---
    unique_words = len(set(words))
    ttr = unique_words / word_count
    # very high TTR over a longer passage can also read as "over-polished";
    # we treat low-to-mid diversity + high uniformity as the AI-leaning combo
    diversity_score = 1.0 - min(ttr * 1.4, 1.0)  # lower diversity -> higher AI-lean score

    # --- 3. stock AI/marketing phrase density ---
    lowered = text.lower()
    phrase_hits = sum(1 for p in AI_STOCK_PHRASES if p in lowered)
    phrase_density = phrase_hits / max(word_count / 100, 1)  # hits per 100 words
    phrase_score = min(phrase_density / 3.0, 1.0)  # saturate at ~3 hits/100 words

    # --- 4. disfluency presence (natural speech marker; ABSENCE raises AI-lean) ---
    disfluency_hits = sum(len(re.findall(pat, lowered)) for pat in DISFLUENCY_MARKERS)
    disfluency_rate = disfluency_hits / max(word_count / 100, 1)
    disfluency_score = max(0.0, 1.0 - min(disfluency_rate / 1.5, 1.0))  # fewer disfluencies -> higher score

    # --- 5. repeated sentence openers (AI text reuses transitional openers) ---
    openers = [ " ".join(_words(s)[:2]) for s in sentences if _words(s) ]
    repeats = len(openers) - len(set(openers)) if openers else 0
    opener_score = min(repeats / max(len(openers), 1) * 2.0, 1.0)

    weights = {
        "sentence_uniformity": (uniformity_score, 0.25),
        "lexical_diversity": (diversity_score, 0.15),
        "stock_phrases": (phrase_score, 0.35),
        "lack_of_disfluency": (disfluency_score, 0.15),
        "repeated_openers": (opener_score, 0.10),
    }
    ai_likelihood = sum(v * w for v, w in weights.values()) * 100

    if ai_likelihood < 35:
        label = "Likely Natural / Spontaneous Speech"
    elif ai_likelihood < 65:
        label = "Mixed / Uncertain"
    else:
        label = "Reads Like a Scripted / AI-Generated Message"

    return {
        "ai_likelihood": round(ai_likelihood, 1),
        "label": label,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "breakdown": {
            "sentence_length_uniformity": round(uniformity_score * 100, 1),
            "low_lexical_diversity": round(diversity_score * 100, 1),
            "stock_phrase_density": round(phrase_score * 100, 1),
            "lack_of_disfluency": round(disfluency_score * 100, 1),
            "repeated_sentence_openers": round(opener_score * 100, 1),
        },
        "matched_stock_phrases": [p for p in AI_STOCK_PHRASES if p in lowered],
    }
