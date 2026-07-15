"""
preprocessor.py — Input Preprocessing Module
=============================================
Handles input normalization, language detection, and intent classification.
These steps run BEFORE matching to improve accuracy and provide metadata.
"""

import re
import unicodedata
from typing import Tuple


# ─────────────────────────────────────────────────────────
# Unicode block ranges for language detection
# ─────────────────────────────────────────────────────────
LANGUAGE_RANGES = {
    "ml": (0x0D00, 0x0D7F),   # Malayalam
    "ta": (0x0B80, 0x0BFF),   # Tamil
    "kn": (0x0C80, 0x0CFF),   # Kannada
    "hi": (0x0900, 0x097F),   # Hindi / Devanagari
}

# ─────────────────────────────────────────────────────────
# Intent classification keyword map
# ─────────────────────────────────────────────────────────
INTENT_KEYWORDS = {
    "ADMISSION": [
        "admission", "apply", "application", "enroll", "eligibility",
        "entrance", "test", "exam", "selection", "offer letter",
        "ಅರ್ಜಿ", "ಪ್ರವೇಶ", "ನೋಂದಾಯಿಸು",
    ],
    "HOSTEL": [
        "hostel", "accommodation", "residence", "devadan", "christ hall",
        "room", "mess", "boys hostel", "girls hostel",
        "ഹോസ്റ്റൽ", "താമസം",
    ],
    "ACADEMICS": [
        "course", "programme", "program", "degree", "curriculum",
        "online", "bba", "bca", "mca", "mcom", "bcom",
        "fee", "fees", "duration", "cost",
        "accreditation", "naac", "nirf", "ranking", "grade",
        "iqac", "quality",
    ],
    "EXAMINATION": [
        "exam", "examination", "timetable", "schedule", "result",
        "answer script", "evaluation", "marks", "question paper",
        "மதிப்பெண்", "தேர்வு", "ரிசல்ட்",
    ],
    "PLACEMENT": [
        "placement", "campus recruit", "interview", "career",
        "aptitude", "hr interview", "group discussion",
        "கேம்பஸ்", "பிளேஸ்மெண்ட்", "நேர்முகத்",
    ],
    "EXCHANGE": [
        "exchange", "study abroad", "international", "partner",
        "outgoing", "incoming", "semester abroad", "ielts", "toefl",
    ],
    "LIBRARY": [
        "library", "book", "journal", "database", "e-resource",
        "shodhganga", "catalogue", "remote access",
        "ಗ್ರಂಥಾಲಯ", "ಸಂಪನ್ಮೂಲ",
    ],
    "STUDENT_LIFE": [
        "student council", "daksh", "event", "fest", "cultural",
        "cicf", "incubation", "consultancy", "startup",
        "ವಿದ್ಯಾರ್ಥಿ", "ಪರಿಷತ್ತು",
    ],
    "RESEARCH": [
        "research", "thesis", "centre", "cell", "innovation",
        "ஆராய்ச்சி", "மையங்கள்",
    ],
}


def normalize_input(text: str) -> str:
    """
    Normalize user input for consistent matching.

    Steps:
        1. Strip leading/trailing whitespace
        2. Collapse multiple spaces into one
        3. Normalize Unicode characters (NFC form)

    Args:
        text: Raw user input string.

    Returns:
        Cleaned, normalized string (case preserved for display;
        matching functions handle case-insensitivity separately).
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = unicodedata.normalize('NFC', text)
    return text


def detect_language(text: str) -> str:
    """
    Detect the dominant language of the input using Unicode block analysis.

    Examines each character's Unicode codepoint and counts how many fall
    within known Indic script ranges. If ≥20% of alphabetic characters
    belong to a single script, that language is returned.

    Args:
        text: The user's input string.

    Returns:
        ISO 639-1 language code: 'ml' (Malayalam), 'ta' (Tamil),
        'kn' (Kannada), 'hi' (Hindi), or 'en' (English/default).
    """
    if not text:
        return "en"

    lang_counts = {lang: 0 for lang in LANGUAGE_RANGES}
    total_alpha = 0

    for char in text:
        if char.isalpha():
            total_alpha += 1
            cp = ord(char)
            for lang, (start, end) in LANGUAGE_RANGES.items():
                if start <= cp <= end:
                    lang_counts[lang] += 1
                    break

    if total_alpha == 0:
        return "en"

    # Find the script with the highest character count
    best_lang = max(lang_counts, key=lang_counts.get)
    best_count = lang_counts[best_lang]

    # Require at least 20% of alphabetic chars to be from that script
    if best_count / total_alpha >= 0.20:
        return best_lang

    return "en"


def classify_intent(text: str) -> str:
    """
    Classify the user's query into a high-level intent category.

    Uses keyword overlap against predefined intent-keyword lists.
    The intent with the most keyword matches wins. If no keywords
    match, returns 'GENERAL'.

    Args:
        text: The user's input string.

    Returns:
        Intent category string, e.g. 'HOSTEL', 'ADMISSION', 'GENERAL'.
    """
    text_lower = text.lower()
    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "GENERAL"

    return max(scores, key=scores.get)


def preprocess(text: str) -> Tuple[str, str, str]:
    """
    Full preprocessing pipeline.

    Runs normalization, language detection, and intent classification
    in sequence and returns all results as a tuple.

    Args:
        text: Raw user input string.

    Returns:
        Tuple of (normalized_text, language_code, intent_category).

    Example:
        >>> preprocess("  What are the hostel fees?  ")
        ('What are the hostel fees?', 'en', 'HOSTEL')
    """
    normalized = normalize_input(text)
    language = detect_language(normalized)
    intent = classify_intent(normalized)
    return normalized, language, intent
