"""
context.py — Conversation Context Manager
==========================================
Tracks conversation state across turns: last topic, matched entry,
and conversation history. Resolves vague follow-up queries by
prepending the last topic to incomplete questions.
"""

import re
from typing import Optional, List, Dict, Any


class ConversationContext:
    """
    Maintains conversation state across multiple chat turns.

    Tracks:
        - Last matched Q&A entry (for follow-up resolution)
        - Last detected intent category
        - Full conversation history (role + content)
        - Count of questions answered

    Attributes:
        last_entry: The most recently matched Q&A entry dict.
        last_intent: The most recently detected intent category.
        history: List of (role, content) tuples.
        questions_answered: Running count of user questions.
    """

    # Patterns that indicate a follow-up / continuation question
    FOLLOWUP_PATTERNS = [
        re.compile(r'^\s*(what about|how about|and)\s+(the\s+)?', re.IGNORECASE),
        re.compile(r'^\s*(tell me more|more info|more details|elaborate)', re.IGNORECASE),
        re.compile(r'^\s*(yes|yeah|yep|sure|ok)\s*[,.]?\s*(tell|what|how)', re.IGNORECASE),
    ]

    # Short vague queries that likely refer to the last topic
    VAGUE_QUERY_PATTERN = re.compile(
        r'^\s*(fees?|cost|duration|process|amenities|facilities|eligibility'
        r'|requirements?|how to apply|details)\s*[?!.]*\s*$',
        re.IGNORECASE
    )

    def __init__(self):
        """Initialize an empty conversation context."""
        self.last_entry: Optional[Dict[str, Any]] = None
        self.last_intent: Optional[str] = None
        self.history: List[Dict[str, str]] = []
        self.questions_answered: int = 0

    def update(
        self,
        user_input: str,
        response: str,
        matched_entry: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None
    ) -> None:
        """
        Update context after a successful exchange.

        Args:
            user_input: What the user asked.
            response: What the bot answered.
            matched_entry: The Q&A entry that was matched (if any).
            intent: The detected intent category.
        """
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        self.questions_answered += 1

        if matched_entry is not None:
            self.last_entry = matched_entry
        if intent is not None:
            self.last_intent = intent

    def resolve_followup(self, user_input: str) -> str:
        """
        Resolve a vague follow-up query by prepending context from the
        last matched entry.

        If the user asks something like "what about the fees?" after
        asking about hostels, this method prepends "hostel" to make
        the query "hostel fees" — which is more likely to match the
        correct pattern.

        Args:
            user_input: The current (potentially vague) user input.

        Returns:
            The enriched user input (or original if no follow-up detected).
        """
        if self.last_entry is None:
            return user_input

        # Skip follow-up resolution for complete questions (6+ words)
        # These are likely standalone queries, not vague follow-ups
        word_count = len(user_input.strip().split())
        if word_count >= 6:
            return user_input

        # Check if this looks like a follow-up
        is_followup = False

        # Check explicit follow-up patterns
        for pattern in self.FOLLOWUP_PATTERNS:
            if pattern.search(user_input):
                is_followup = True
                break

        # Check for vague single-word/short queries
        if not is_followup and self.VAGUE_QUERY_PATTERN.match(user_input):
            is_followup = True

        if not is_followup:
            return user_input

        # Extract topic keywords from the last matched entry
        last_question = (
            self.last_entry.get('question_english', '')
            or self.last_entry.get('question', '')
        )

        # Extract the main topic noun from the last question
        topic_words = []
        important_words = re.findall(r'\b\w{4,}\b', last_question.lower())
        stop_words = {
            'what', 'which', 'where', 'when', 'does', 'have',
            'that', 'this', 'with', 'from', 'about', 'available',
            'christ', 'university', 'students', 'international',
        }
        for word in important_words:
            if word not in stop_words:
                topic_words.append(word)
                if len(topic_words) >= 2:
                    break

        if topic_words:
            topic = ' '.join(topic_words)
            # Prepend topic to the follow-up query
            enriched = f"{topic} {user_input}"
            return enriched

        return user_input

    def is_repeated_question(self, entry: Dict[str, Any]) -> bool:
        """
        Check if the user is asking the same question as last time.

        Args:
            entry: The Q&A entry that was just matched.

        Returns:
            True if this entry was the last one answered.
        """
        if self.last_entry is None:
            return False
        return str(entry.get('id', '')) == str(self.last_entry.get('id', ''))

    def get_session_summary(self) -> str:
        """
        Generate a summary of the conversation session.

        Returns:
            A formatted string summarizing the session.
        """
        if self.questions_answered == 0:
            return "No questions were asked in this session."

        SUMMARY_STOP_WORDS = {
            'what', 'which', 'where', 'when', 'does', 'have', 'that',
            'this', 'with', 'from', 'about', 'tell', 'much', 'many',
            'christ', 'university', 'online', 'process', 'available',
            'please', 'know', 'give', 'explain', 'describe',
        }
        topics = set()
        for msg in self.history:
            if msg["role"] == "user":
                # Extract meaningful topic words from user messages
                words = re.findall(r'\b\w{4,}\b', msg["content"].lower())
                filtered = [w for w in words if w not in SUMMARY_STOP_WORDS]
                topics.update(filtered[:2])

        topic_str = ', '.join(sorted(topics)[:5]) if topics else 'various topics'
        return (
            f"📊 Session Summary: You asked {self.questions_answered} "
            f"question(s) about {topic_str}."
        )
