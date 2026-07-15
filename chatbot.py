"""
🤖 CHRIST University Regex Chatbot — Streamlit App
CIA 1 — Advanced NLP

Features:
    - Advanced regex matching with named capture groups
    - Dynamic entity extraction and templated responses
    - Confidence scoring with color-coded badges
    - "Did you mean?" suggestions for unmatched queries
    - Follow-up question suggestions
    - Conversation context tracking
    - Multilingual support (English, Malayalam, Tamil, Kannada)
    - Intent classification tags

Run with: streamlit run chatbot.py
"""

import streamlit as st
import os
import glob
import sys
import streamlit.components.v1 as components

# Add project root to path so engine module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import ChatEngine
from engine.response_builder import format_confidence_label


# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CHRIST University Chatbot",
    page_icon="🤖",
    layout="centered",
)


# ─────────────────────────────────────────────────────────
# Custom CSS for a premium chat UI
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #fff !important;
    }

    /* Chat input styling */
    .stChatInput > div {
        border-radius: 25px !important;
    }

    /* Stats badge */
    .stats-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }

    /* Confidence badges */
    .confidence-high {
        display: inline-block;
        background: #1a7a3a;
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    .confidence-mid {
        display: inline-block;
        background: #b8860b;
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    .confidence-low {
        display: inline-block;
        background: #a33;
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 500;
    }

    /* Intent tag */
    .intent-tag {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-left: 0.3rem;
    }

    /* Language tag */
    .lang-tag {
        display: inline-block;
        background: rgba(118, 75, 162, 0.15);
        color: #764ba2;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-left: 0.3rem;
    }

    /* Divider */
    .subtle-divider {
        border: none;
        border-top: 1px solid #333;
        margin: 1rem 0;
    }

    /* Metadata row */
    .meta-row {
        margin-top: 0.5rem;
        display: flex;
        gap: 0.3rem;
        flex-wrap: wrap;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Language display names
# ─────────────────────────────────────────────────────────
LANGUAGE_NAMES = {
    "en": "English",
    "ml": "Malayalam",
    "ta": "Tamil",
    "kn": "Kannada",
    "hi": "Hindi",
}


# ─────────────────────────────────────────────────────────
# Initialize ChatEngine (cached)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    """Create and cache the ChatEngine instance."""
    return ChatEngine()


engine = get_engine()


# ─────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm the CHRIST University Chatbot. "
                "I can help you with information about:\n\n"
                "• Online courses & admissions\n"
                "• Hostel facilities & fees\n"
                "• Accreditations & rankings\n"
                "• Placements & career guidance\n"
                "• Library resources\n"
                "• Student exchange programs\n\n"
                "Ask me anything, or type **help** to see all topics!"
            ),
            "meta": None,
        }
    ]

if "engine_context_initialized" not in st.session_state:
    st.session_state.engine_context_initialized = True
    # The engine has its own context; reset it for a fresh session
    engine.reset_context()


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 CHRIST Chatbot")
    st.markdown("**CIA 1 — Advanced NLP**")
    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

    st.markdown(
        f"<span class='stats-badge'>📚 {len(engine.qa_data)} Q&As loaded</span>"
        f"<span class='stats-badge'>🧠 {len(engine.regex_matcher.compiled_patterns)} patterns</span>",
        unsafe_allow_html=True,
    )

    # List loaded files
    json_files = sorted(glob.glob(os.path.join('qa_data', '*.json')))
    if json_files:
        st.markdown("### 📁 Data Files")
        for fp in json_files:
            st.markdown(f"• `{os.path.basename(fp)}`")

    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

    st.markdown("### 📋 Available Categories")
    cat_icon_map = {
        "ADMISSION": "🎓", "HOSTEL": "🏠", "ACADEMICS": "📚",
        "EXAMINATION": "📝", "PLACEMENT": "💼", "STUDENT_EXCHANGE": "🌍",
        "LIBRARY": "📖", "STUDENT_LIFE": "🎭", "RESEARCH": "🔬",
    }
    # Collect unique categories
    categories = sorted(set(entry.get('category', 'OTHER') for entry in engine.qa_data))
    # Display each category with icon
    for cat in categories:
        icon = cat_icon_map.get(cat, "📌")
        display_name = cat.replace("_", " ").title()
        if display_name == "Exchange":
            display_name = "Student Exchange"
        st.markdown(f"{icon} **{display_name}**")

    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

    st.markdown("### 💡 Try asking:")
    st.markdown("- *What is the fee for BCA?*")
    st.markdown("- *boys hostel fees*")
    st.markdown("- *engineering exchange program*")
    st.markdown("- *NAAC ranking*")
    st.markdown("- *previous question papers*")
    st.markdown("- *What is Daksh?*")

    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        engine.reset_context()
        st.rerun()


# ─────────────────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 CHRIST University Chatbot</h1>
    <p>Powered by Regular Expressions • Dynamic Entity Extraction • Multilingual NLP</p>
</div>
""", unsafe_allow_html=True)


def render_metadata_badges(meta):
    """Render confidence, intent, and language badges as HTML."""
    if meta is None:
        return ""

    confidence = meta.get('confidence', 0)
    pct = int(confidence * 100)
    match_type = meta.get('match_type', 'none')
    intent = meta.get('intent', 'GENERAL')
    language = meta.get('language', 'en')
    lang_name = LANGUAGE_NAMES.get(language, language.upper())

    # Confidence badge class
    if confidence >= 0.85:
        conf_class = "confidence-high"
        conf_text = f"🟢 {pct}% Regex"
    elif confidence >= 0.50:
        conf_class = "confidence-mid"
        conf_text = f"🟡 {pct}% Keyword"
    elif confidence > 0:
        conf_class = "confidence-low"
        conf_text = f"🔴 {pct}%"
    else:
        conf_class = "confidence-low"
        conf_text = "❌ No Match"

    # Only show non-trivial intents
    intent_html = ""
    if intent not in ("GREETING", "THANKS", "GOODBYE", "HELP", "GENERAL"):
        intent_html = f"<span class='intent-tag'>🏷️ {intent}</span>"

    # Only show language for non-English
    lang_html = ""
    if language != "en":
        lang_html = f"<span class='lang-tag'>🌐 {lang_name}</span>"

    return (
        f"<div class='meta-row'>"
        f"<span class='{conf_class}'>{conf_text}</span>"
        f"{intent_html}{lang_html}"
        f"</div>"
    )


# Display chat messages
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        meta = message.get("meta")
        if meta:
            st.markdown(
                render_metadata_badges(meta),
                unsafe_allow_html=True,
            )

# Chat input
if prompt := st.chat_input("Ask me about CHRIST University..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "meta": None,
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Get bot response
    result = engine.respond(prompt)

    # Build the response content
    response_content = result['response']

    # Add follow-up suggestions
    followups = result.get('followups', [])
    if followups:
        response_content += "\n\n---\n💡 **You might also want to know:**"
        for fq in followups[:3]:
            response_content += f"\n- {fq}"

    # Build metadata
    meta = {
        'confidence': result['confidence'],
        'match_type': result['match_type'],
        'intent': result['intent'],
        'language': result['language'],
    }

    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_content,
        "meta": meta,
    })
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response_content)
        st.markdown(
            render_metadata_badges(meta),
            unsafe_allow_html=True,
        )

    # Auto-scroll to the latest message
    components.html("""
    <script>
        window.parent.document.querySelector('section.main').scrollTo({
            top: window.parent.document.querySelector('section.main').scrollHeight,
            behavior: 'smooth'
        });
    </script>
    """, height=0)
