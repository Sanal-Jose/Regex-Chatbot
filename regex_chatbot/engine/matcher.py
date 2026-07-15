"""
matcher.py — Pattern Matching Engine
=====================================
Two-level matching: primary regex matching with confidence scoring,
and keyword-based fallback with IDF-weighted scoring.
Also provides "did you mean?" similar question suggestions.
"""

import re
import math
from typing import Optional, List, Tuple, Dict, Any


class RegexMatcher:
    """
    Primary matching engine using compiled regex patterns.

    Compiles all patterns from Q&A data once, then matches user input
    against them. Returns the best match ranked by match specificity
    (longest match wins). Confidence is calculated based on match
    coverage relative to user input length.

    Attributes:
        compiled_patterns: List of (compiled_regex, qa_entry) tuples.
    """

    def __init__(self, qa_data: List[Dict[str, Any]]):
        """
        Compile all regex patterns from Q&A entries.

        Args:
            qa_data: List of Q&A entry dicts, each with a 'pattern' key.
        """
        self.compiled_patterns: List[Tuple[re.Pattern, Dict]] = []
        self.errors: List[str] = []

        for entry in qa_data:
            pattern_str = entry.get('pattern', '')
            if not pattern_str:
                continue
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                self.compiled_patterns.append((compiled, entry))
            except re.error as e:
                self.errors.append(
                    f"Invalid regex in Q{entry.get('id', '?')}: {e}"
                )

    def match(self, user_input: str) -> Optional[Tuple[Dict, re.Match, float]]:
        """
        Match user input against all compiled regex patterns.

        Scoring algorithm:
            - Each pattern is searched (not full-matched) against input
            - Score = length of the matched substring
            - Longest match = most specific = best match
            - Confidence = match_length / input_length, scaled to 85-100%

        Args:
            user_input: The normalized user query string.

        Returns:
            Tuple of (qa_entry, match_object, confidence) for the best
            match, or None if no pattern matches.
        """
        best_match = None
        best_score = 0
        best_match_obj = None

        for compiled_pattern, entry in self.compiled_patterns:
            match = compiled_pattern.search(user_input)
            if match:
                score = len(match.group())
                if score > best_score:
                    best_score = score
                    best_match = entry
                    best_match_obj = match

        if best_match is None:
            return None

        # Calculate confidence: how much of the input was captured
        input_len = max(len(user_input.strip()), 1)
        coverage = best_score / input_len
        # Scale to 85-100% range for regex matches
        confidence = 0.85 + (coverage * 0.15)
        confidence = min(confidence, 1.0)

        return (best_match, best_match_obj, confidence)


class KeywordMatcher:
    """
    Fallback matching engine using keyword overlap scoring.

    When regex patterns fail to match, this engine tokenizes the user
    input, removes stop words, and counts keyword overlaps against the
    searchable text of each Q&A entry (question, question_english,
    sample_queries). Uses IDF weighting so rare keywords score higher.

    Attributes:
        qa_data: The full list of Q&A entries.
        idf_scores: Pre-computed IDF scores for all terms in the corpus.
    """

    # Common English stop words to filter out
    STOP_WORDS = frozenset({
        'what', 'is', 'the', 'a', 'an', 'are', 'how', 'can', 'i',
        'do', 'does', 'tell', 'me', 'about', 'in', 'at', 'of',
        'for', 'to', 'and', 'or', 'it', 'this', 'that', 'there',
        'have', 'has', 'be', 'been', 'was', 'were', 'will', 'would',
        'could', 'should', 'may', 'might', 'with', 'from', 'on', 'by',
        'please', 'like', 'want', 'need', 'know', 'get', 'give',
    })

    def __init__(self, qa_data: List[Dict[str, Any]]):
        """
        Initialize with Q&A data and pre-compute IDF scores.

        Args:
            qa_data: List of Q&A entry dicts.
        """
        self.qa_data = qa_data
        self.idf_scores = self._compute_idf()

    def _get_searchable_text(self, entry: Dict) -> str:
        """Extract and combine all searchable text fields from a Q&A entry."""
        texts = [
            entry.get('question', '').lower(),
            entry.get('question_english', '').lower(),
        ]
        for sq in entry.get('sample_queries', []):
            texts.append(sq.lower())
        return ' '.join(texts)

    def _compute_idf(self) -> Dict[str, float]:
        """
        Compute Inverse Document Frequency for all terms.

        IDF(term) = log(N / df(term)), where N is the total number of
        Q&A entries and df is the count of entries containing the term.
        Higher IDF = rarer term = more discriminative.
        """
        n_docs = len(self.qa_data)
        if n_docs == 0:
            return {}

        # Count document frequency for each term
        df = {}
        for entry in self.qa_data:
            text = self._get_searchable_text(entry)
            unique_words = set(re.findall(r'\w+', text))
            for word in unique_words:
                df[word] = df.get(word, 0) + 1

        # Compute IDF
        idf = {}
        for word, freq in df.items():
            idf[word] = math.log(n_docs / freq) + 1.0  # +1 smoothing
        return idf

    def match(self, user_input: str) -> Optional[Tuple[Dict, float]]:
        """
        Match user input against Q&A entries using IDF-weighted keywords.

        Algorithm:
            1. Tokenize user input and remove stop words
            2. For each Q&A entry, compute IDF-weighted keyword overlap
            3. Return the highest-scoring entry if score meets threshold

        Args:
            user_input: The normalized user query string.

        Returns:
            Tuple of (qa_entry, confidence) or None if no match.
            Confidence is scaled to 50-84% range for keyword matches.
        """
        user_words = re.findall(r'\w+', user_input.lower())
        user_keywords = [w for w in user_words if w not in self.STOP_WORDS]

        # Fallback: if all words were stop words, use all words
        if not user_keywords:
            user_keywords = user_words

        if not user_keywords:
            return None

        best_match = None
        best_score = 0.0
        max_possible_score = sum(
            self.idf_scores.get(kw, 1.0) for kw in user_keywords
        )

        for entry in self.qa_data:
            combined_text = self._get_searchable_text(entry)

            score = 0.0
            matches_count = 0
            for keyword in user_keywords:
                if keyword in combined_text:
                    score += self.idf_scores.get(keyword, 1.0)
                    matches_count += 1

            if score > best_score:
                best_score = score
                best_match = entry

        # Require at least 2 keyword matches (or 1 if query is a single word)
        min_matches = 1 if len(user_keywords) <= 1 else 2

        if best_match is None or best_score == 0:
            return None

        # Re-count matches for the best match to check threshold
        combined_text = self._get_searchable_text(best_match)
        actual_matches = sum(
            1 for kw in user_keywords if kw in combined_text
        )

        if actual_matches < min_matches:
            return None

        # Scale confidence to 50-84% range
        if max_possible_score > 0:
            raw_confidence = best_score / max_possible_score
        else:
            raw_confidence = 0.5
        confidence = 0.50 + (raw_confidence * 0.34)
        confidence = min(confidence, 0.84)

        return (best_match, confidence)


def find_similar_questions(
    user_input: str,
    qa_data: List[Dict[str, Any]],
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Find the most similar questions for "Did you mean?" suggestions.

    Uses simple word overlap scoring (without stop word filtering)
    to find Q&A entries that are loosely related to the user's query.

    Args:
        user_input: The user's query that failed to match.
        qa_data: Full list of Q&A entries.
        top_n: Number of suggestions to return.

    Returns:
        List of up to top_n Q&A entries, ranked by similarity.
    """
    user_words = set(re.findall(r'\w+', user_input.lower()))

    if not user_words:
        return []

    scored = []
    for entry in qa_data:
        texts = [
            entry.get('question', '').lower(),
            entry.get('question_english', '').lower(),
        ]
        for sq in entry.get('sample_queries', []):
            texts.append(sq.lower())
        combined = ' '.join(texts)
        entry_words = set(re.findall(r'\w+', combined))

        # Jaccard-like similarity
        if not entry_words:
            continue
        intersection = len(user_words & entry_words)
        union = len(user_words | entry_words)
        if intersection > 0:
            scored.append((intersection / union, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:top_n]]
