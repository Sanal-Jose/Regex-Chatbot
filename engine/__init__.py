"""
engine/ — CHRIST University Regex Chatbot Engine
=================================================
A modular, rule-based chatbot engine using Regular Expressions.

Architecture:
    ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
    │ User Input   │────▶│ Preprocessor │────▶│ Context Resolver │
    └─────────────┘     └──────────────┘     └──────────────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │ Regex Matcher   │
                                              │ (Primary, 85%+) │
                                              └───────┬────────┘
                                                      │ miss?
                                              ┌───────▼────────┐
                                              │ Keyword Matcher │
                                              │ (Fallback, 50%) │
                                              └───────┬────────┘
                                                      │ miss?
                                              ┌───────▼────────┐
                                              │ Similar Qs      │
                                              │ ("Did you mean?")│
                                              └───────┬────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │ Response Builder │
                                              │ (Dynamic/Static)│
                                              └────────────────┘

Components:
    - preprocessor: Input normalization, language detection, intent classification
    - matcher: RegexMatcher (primary) + KeywordMatcher (fallback)
    - response_builder: Dynamic entity-based + static response rendering
    - context: Conversation memory and follow-up resolution

Usage:
    >>> from engine import ChatEngine
    >>> bot = ChatEngine()
    >>> result = bot.respond("What are the hostel fees?")
    >>> print(result['response'])
    >>> print(result['confidence'])
"""

import re
import json
import os
import glob
from typing import Dict, Any, List, Optional

from .preprocessor import preprocess, normalize_input, detect_language, classify_intent
from .matcher import RegexMatcher, KeywordMatcher, find_similar_questions
from .response_builder import (
    build_response, get_followup_suggestions, format_confidence_label
)
from .context import ConversationContext


class ChatEngine:
    """
    Main chatbot engine — single entry point for all chatbot operations.

    Orchestrates preprocessing, matching, response building, and context
    management. Designed to be used by both the Streamlit app and the
    Jupyter notebook.

    Attributes:
        qa_data: Full list of loaded Q&A entries.
        regex_matcher: Primary regex-based matching engine.
        keyword_matcher: Fallback keyword-based matching engine.
        context: Conversation context manager.
    """

    # ── Built-in conversational patterns ──
    GREETING_PATTERN = re.compile(
        r'^\s*(hi|hello|hey|good\s*morning|good\s*afternoon|good\s*evening'
        r'|namaste|hola|howdy|sup)\s*[!.?]*\s*$',
        re.IGNORECASE
    )
    THANKS_PATTERN = re.compile(
        r'(thank|thanks|thank\s*you|dhanyavaad|nandri|dhanyavad)',
        re.IGNORECASE
    )
    HELP_PATTERN = re.compile(
        r'^\s*(help|what can you do|commands|menu|topics)\s*[?!.]*\s*$',
        re.IGNORECASE
    )
    GOODBYE_PATTERN = re.compile(
        r'^\s*(quit|exit|bye|goodbye|stop|see\s*you|close)\s*[!.?]*\s*$',
        re.IGNORECASE
    )

    def __init__(self, qa_folder: str = 'qa_data'):
        """
        Initialize the chat engine: load data, compile patterns, set up matchers.

        Args:
            qa_folder: Path to the folder containing Q&A JSON files.
        """
        self.qa_data = self._load_qa_data(qa_folder)
        self.regex_matcher = RegexMatcher(self.qa_data)
        self.keyword_matcher = KeywordMatcher(self.qa_data)
        self.context = ConversationContext()

    def _load_qa_data(self, qa_folder: str) -> List[Dict[str, Any]]:
        """
        Load all Q&A JSON files from the specified folder.

        Scans for *.json files, parses each, and concatenates all entries
        into a single list. Silently skips files that fail to parse.

        Args:
            qa_folder: Path to the Q&A data directory.

        Returns:
            Combined list of Q&A entry dicts from all JSON files.
        """
        all_qa: List[Dict[str, Any]] = []
        json_files = sorted(glob.glob(os.path.join(qa_folder, '*.json')))

        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_qa.extend(data)
            except Exception:
                pass

        return all_qa

    def reset_context(self) -> None:
        """Reset the conversation context (e.g., when clearing chat)."""
        self.context = ConversationContext()

    def respond(self, user_input: str) -> Dict[str, Any]:
        """
        Generate a full response to a user query.

        This is the main entry point. It runs the full pipeline:
            1. Preprocess (normalize, detect language, classify intent)
            2. Check for built-in patterns (greeting, thanks, help, goodbye)
            3. Resolve follow-ups using conversation context
            4. Try regex matching (primary)
            5. Try keyword matching (fallback)
            6. Generate "did you mean?" suggestions if nothing matched
            7. Build the response (dynamic or static)
            8. Generate follow-up suggestions
            9. Update conversation context

        Args:
            user_input: Raw user input string.

        Returns:
            Dict with keys:
                - response (str): The bot's response text
                - confidence (float): Confidence score 0.0-1.0
                - confidence_label (str): Human-readable confidence
                - language (str): Detected language code
                - intent (str): Classified intent category
                - match_type (str): 'regex', 'keyword', 'builtin', or 'none'
                - followups (list[str]): Suggested follow-up questions
                - suggestions (list[str]): "Did you mean?" suggestions
                - matched_entry (dict|None): The matched Q&A entry
        """
        # ── Step 1: Preprocess ──
        normalized, language, intent = preprocess(user_input)

        result = {
            'response': '',
            'confidence': 0.0,
            'confidence_label': '',
            'language': language,
            'intent': intent,
            'match_type': 'none',
            'followups': [],
            'suggestions': [],
            'matched_entry': None,
        }

        # ── Step 2: Built-in conversational patterns ──
        if self.GREETING_PATTERN.match(normalized):
            result['response'] = (
                "Hello! 👋 I'm the CHRIST University Chatbot. I can help you "
                "with information about online courses, admissions, hostel "
                "facilities, accreditations, placements, library resources, "
                "and student exchange programs. Ask me anything!"
            )
            result['confidence'] = 1.0
            result['match_type'] = 'builtin'
            result['confidence_label'] = format_confidence_label(1.0)
            result['intent'] = 'GREETING'
            self.context.update(user_input, result['response'], intent='GREETING')
            return result

        if self.THANKS_PATTERN.search(normalized):
            result['response'] = (
                "You're welcome! 😊 Feel free to ask if you have any more "
                "questions about CHRIST University."
            )
            result['confidence'] = 1.0
            result['match_type'] = 'builtin'
            result['confidence_label'] = format_confidence_label(1.0)
            result['intent'] = 'THANKS'
            self.context.update(user_input, result['response'], intent='THANKS')
            return result

        if self.GOODBYE_PATTERN.match(normalized):
            summary = self.context.get_session_summary()
            result['response'] = f"Goodbye! 👋 Have a great day!\n\n{summary}"
            result['confidence'] = 1.0
            result['match_type'] = 'builtin'
            result['confidence_label'] = format_confidence_label(1.0)
            result['intent'] = 'GOODBYE'
            return result

        if self.HELP_PATTERN.match(normalized):
            topics = []
            for entry in self.qa_data:
                q_english = entry.get('question_english', '')
                q_native = entry.get('question', '')
                display_q = q_english if q_english else q_native
                if display_q:
                    topics.append(f"• {display_q}")
            result['response'] = (
                "I can answer questions about:\n\n"
                + "\n".join(topics)
                + "\n\nJust type your question naturally!"
            )
            result['confidence'] = 1.0
            result['match_type'] = 'builtin'
            result['confidence_label'] = format_confidence_label(1.0)
            result['intent'] = 'HELP'
            self.context.update(user_input, result['response'], intent='HELP')
            return result

        # ── Step 3: Resolve follow-ups ──
        enriched_input = self.context.resolve_followup(normalized)

        # ── Step 4: Regex matching (primary) ──
        regex_result = self.regex_matcher.match(enriched_input)
        if regex_result:
            entry, match_obj, confidence = regex_result
            response_text = build_response(entry, match_obj, enriched_input, language)

            # Check for repeated question
            if self.context.is_repeated_question(entry):
                response_text = (
                    "As I mentioned earlier:\n\n" + response_text
                )

            result['response'] = response_text
            result['confidence'] = confidence
            result['confidence_label'] = format_confidence_label(confidence)
            result['match_type'] = 'regex'
            result['matched_entry'] = entry
            result['followups'] = get_followup_suggestions(
                entry, self.qa_data
            )
            self.context.update(
                user_input, response_text, matched_entry=entry, intent=intent
            )
            return result

        # ── Step 5: Keyword matching (fallback) ──
        keyword_result = self.keyword_matcher.match(enriched_input)
        if keyword_result:
            entry, confidence = keyword_result
            response_text = build_response(entry, None, enriched_input, language)

            if self.context.is_repeated_question(entry):
                response_text = (
                    "As I mentioned earlier:\n\n" + response_text
                )

            result['response'] = response_text
            result['confidence'] = confidence
            result['confidence_label'] = format_confidence_label(confidence)
            result['match_type'] = 'keyword'
            result['matched_entry'] = entry
            result['followups'] = get_followup_suggestions(
                entry, self.qa_data
            )
            self.context.update(
                user_input, response_text, matched_entry=entry, intent=intent
            )
            return result

        # ── Step 6: No match — generate suggestions ──
        similar = find_similar_questions(
            enriched_input, self.qa_data, top_n=3
        )
        suggestion_texts = []
        for s in similar:
            q = s.get('question_english') or s.get('question', '')
            if q:
                suggestion_texts.append(q)

        if suggestion_texts:
            result['response'] = (
                "🤔 I'm not sure about that. Did you mean one of these?\n\n"
                + "\n".join(f"• {s}" for s in suggestion_texts)
                + "\n\nOr type **help** to see all available topics."
            )
            result['suggestions'] = suggestion_texts
        else:
            result['response'] = (
                "🤔 I'm sorry, I couldn't find an answer to your question. "
                "Please try rephrasing, or type **help** to see the topics "
                "I can assist with."
            )

        result['confidence'] = 0.0
        result['confidence_label'] = format_confidence_label(0.0)
        self.context.update(user_input, result['response'], intent=intent)
        return result
