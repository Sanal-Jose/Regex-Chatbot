"""
response_builder.py — Dynamic Response Generation
===================================================
Builds responses using entity extraction from regex named groups,
template rendering, and follow-up suggestion generation.
"""

import re
from typing import Optional, List, Dict, Any


def build_response(
    entry: Dict[str, Any],
    match_obj: Optional[re.Match],
    user_input: str,
    language: str = 'en'
) -> str:
    """
    Build a dynamic response from a Q&A entry and regex match.

    Strategy:
        1. If the entry has an 'answer_template' and 'data' dict, try to
           extract a named group from the regex match and look it up in
           the data dict. Render the template with the specific data.
        2. If no template or no named group extracted, check for
           'conditional_answers' — a dict mapping captured values to
           specific answer variants.
        3. Fall back to the static 'answer' field.

    For bilingual entries (those with 'question_english'), the response
    adapts to the user's detected language:
        - English user → shows only the English portion of the answer
        - Non-English user → shows native language first, then English translation

    Args:
        entry: The matched Q&A entry dict.
        match_obj: The regex Match object (may be None for keyword matches).
        user_input: The original user input (for additional entity extraction).
        language: Detected language code ('en', 'ml', 'ta', 'kn', 'hi').

    Returns:
        The formatted response string.
    """
    # ── Strategy 1: Template + Data rendering ──
    template = entry.get('answer_template')
    data = entry.get('data')

    if template and data and match_obj:
        # Try to extract named groups from the regex match
        groups = match_obj.groupdict()
        # Find the first non-None named group
        extracted = {}
        for key, value in groups.items():
            if value is not None:
                extracted[key] = value.strip().lower()

        if extracted:
            # Look up in data dict
            for key, value in extracted.items():
                normalized_val = value.upper()
                if normalized_val in data:
                    item = data[normalized_val]
                    try:
                        # Merge: item data takes precedence over group key
                        fmt_args = {key: value.upper()}
                        fmt_args.update(item)
                        return template.format(**fmt_args)
                    except (KeyError, IndexError, TypeError):
                        pass
                # Try case-insensitive lookup
                for data_key, item in data.items():
                    if data_key.lower() == value:
                        try:
                            fmt_args = {key: data_key}
                            fmt_args.update(item)
                            return template.format(**fmt_args)
                        except (KeyError, IndexError, TypeError):
                            pass

    # ── Strategy 2: Conditional answers ──
    conditional = entry.get('conditional_answers')
    if conditional:
        # First try: extract from named regex groups
        if match_obj:
            groups = match_obj.groupdict()
            for key, value in groups.items():
                if value is not None:
                    val_lower = value.strip().lower()
                    # Check for matches in conditional answers
                    for cond_key, cond_answer in conditional.items():
                        if val_lower in cond_key.lower().split('|'):
                            return cond_answer
                    # Also try the group name as the condition key
                    if key in conditional:
                        return conditional[key]

        # Fallback: scan the user_input for conditional keywords
        input_lower = user_input.lower()
        for cond_key, cond_answer in conditional.items():
            keywords = cond_key.lower().split('|')
            for kw in keywords:
                if kw.strip() in input_lower:
                    return cond_answer

    # ── Strategy 3: Plain static answer ──
    answer = entry.get('answer', 'Sorry, no answer available.')

    # Handle bilingual entries based on the user's detected language
    q_english = entry.get('question_english', '')
    if q_english:
        # The answer contains both native and English portions
        # separated by "\n\n---\n\n" with the English part prefixed by "(English)"
        return _format_bilingual_answer(answer, q_english, language)

    return answer


def _format_bilingual_answer(answer: str, q_english: str, language: str) -> str:
    """
    Format a bilingual answer based on the user's detected language.

    Bilingual answers may follow two formats in the JSON data:
        Format A: [Native answer]\n\n---\n\n(English) [English answer]
        Format B: [Native answer]\n\n(English) [English answer]

    Rules:
        - If the user asked in English → return ONLY the English portion
        - If the user asked in a non-English language → return the native
          language answer first, followed by the English translation with
          a 🌐 header

    Args:
        answer: The full bilingual answer string from the Q&A entry.
        q_english: The English version of the question (for the header).
        language: Detected language code of the user's input.

    Returns:
        Formatted answer string appropriate for the user's language.
    """
    native_part = None
    english_part = None

    # Try Format A: split on "---" divider
    if '\n\n---\n\n' in answer:
        parts = answer.split('\n\n---\n\n', 1)
        native_part = parts[0].strip()
        english_part = parts[1].strip()
    # Try Format B: split on "(English)" marker
    elif '\n\n(English)' in answer:
        idx = answer.index('\n\n(English)')
        native_part = answer[:idx].strip()
        english_part = answer[idx:].strip()

    if native_part is not None and english_part is not None:
        # Remove the "(English) " prefix if present
        if english_part.startswith('(English)'):
            english_part = english_part[len('(English)'):].strip()

        if language == 'en':
            # English user → show only the English answer
            return english_part
        else:
            # Non-English user → show native first, then English translation
            return (
                f"{native_part}\n\n"
                f"---\n\n"
                f"**🌐 {q_english}**\n\n"
                f"{english_part}"
            )
    else:
        # No split found — show the full answer with/without header
        if language == 'en':
            return answer
        else:
            return f"**🌐 {q_english}**\n\n{answer}"


def get_followup_suggestions(
    current_entry: Dict[str, Any],
    qa_data: List[Dict[str, Any]],
    max_suggestions: int = 3
) -> List[str]:
    """
    Generate follow-up question suggestions based on the current answer.

    Uses two strategies:
        1. Explicit 'related_ids' — if the current entry specifies related
           question IDs, use those.
        2. Category matching — find other entries in the same category.

    Args:
        current_entry: The Q&A entry that was just matched.
        qa_data: Full list of Q&A entries.
        max_suggestions: Maximum number of suggestions to return.

    Returns:
        List of question strings the user might want to ask next.
    """
    suggestions = []
    current_id = str(current_entry.get('id', ''))
    current_category = current_entry.get('category', '')

    # Strategy 1: Explicit related IDs
    related_ids = current_entry.get('related_ids', [])
    if related_ids:
        for rid in related_ids:
            if len(suggestions) >= max_suggestions:
                break
            for entry in qa_data:
                if str(entry.get('id', '')) == str(rid):
                    q = entry.get('question_english') or entry.get('question', '')
                    if q and q not in suggestions:
                        suggestions.append(q)
                    break

    # Strategy 2: Same category
    if len(suggestions) < max_suggestions and current_category:
        for entry in qa_data:
            if len(suggestions) >= max_suggestions:
                break
            if (str(entry.get('id', '')) != current_id
                    and entry.get('category', '') == current_category):
                q = entry.get('question_english') or entry.get('question', '')
                if q and q not in suggestions:
                    suggestions.append(q)

    return suggestions[:max_suggestions]


def format_confidence_label(confidence: float) -> str:
    """
    Format a confidence score into a human-readable label with emoji.

    Args:
        confidence: Float between 0.0 and 1.0.

    Returns:
        Formatted string like '🟢 95% (Regex Match)'.
    """
    pct = int(confidence * 100)
    if confidence >= 0.85:
        return f"🟢 {pct}% (Regex Match)"
    elif confidence >= 0.60:
        return f"🟡 {pct}% (Keyword Match)"
    else:
        return f"🔴 {pct}% (Low Confidence)"
