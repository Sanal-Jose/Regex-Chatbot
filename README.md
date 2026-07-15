# 🤖 CHRIST University Regex Chatbot

**CIA 1 — Advanced NLP | M.Tech Data Science**

A modular, rule-based chatbot that uses **advanced Regular Expressions** with named capture groups, lookaheads, and dynamic entity extraction to match user queries and provide intelligent, context-aware responses about CHRIST University.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Advanced Regex Patterns** | Lookaheads, named groups `(?P<name>)`, non-capturing groups, typo tolerance, anchors |
| **Dynamic Entity Extraction** | Extracts programme names, gender, department from queries for specific answers |
| **Template-based Responses** | "What is the fee for BCA?" → returns BCA-specific fee and duration |
| **Conditional Answers** | "boys hostel fees" → returns boys-only info; "girls hostel" → girls-only |
| **Multilingual Support** | English, Malayalam, Tamil, Kannada with Unicode-aware patterns |
| **Two-level Matching** | Primary regex (85-100% confidence) + IDF-weighted keyword fallback (50-84%) |
| **Confidence Scoring** | Every response shows match confidence with color-coded badges |
| **Conversation Context** | Tracks last topic, resolves vague follow-ups like "what about the fees?" |
| **"Did you mean?" Suggestions** | Jaccard similarity-based suggestions when no match is found |
| **Follow-up Suggestions** | Related questions suggested after each answer |
| **Modular Architecture** | 5-module engine: preprocessor → matcher → response_builder → context → orchestrator |

---

## 📁 Project Structure

```
.
├── engine/                      ← Core chatbot engine (modular)
│   ├── __init__.py              ← ChatEngine orchestrator class
│   ├── preprocessor.py          ← Input normalization, language detection, intent classification
│   ├── matcher.py               ← RegexMatcher (primary) + KeywordMatcher (fallback)
│   ├── response_builder.py      ← Dynamic response rendering, follow-ups, confidence labels
│   └── context.py               ← Conversation memory, follow-up resolution
├── qa_data/                     ← Q&A data (mapped to team members)
│   ├── qa_1.json                ← Defitha (IDs D1–D15, English & Tamil)
│   ├── qa_2.json                ← Evengiline (IDs E1–E15, English & Kannada)
│   ├── qa_3.json                ← Sanal (IDs 1–17, English & Malayalam)
│   └── qa_4.json                ← Nasreen (IDs N1–N15, English & Hindi)
├── regex_chatbot.ipynb          ← Jupyter Notebook (primary evaluation artifact)
├── chatbot.py                   ← Streamlit Web App
├── requirements.txt             ← Dependencies
└── README.md                    ← This file
```

---

## 🏗️ Architecture

```
User Input → Preprocessor → Context Resolver → Regex Matcher → Response Builder
                                                    ↓ (miss)
                                              Keyword Matcher
                                                    ↓ (miss)
                                              Similar Questions
                                              ("Did you mean?")
```

**Preprocessor:** Normalizes input, detects language (Unicode block analysis), classifies intent (keyword overlap).

**Regex Matcher:** Compiles all patterns once, scores by longest match (most specific wins). Confidence: 85-100%.

**Keyword Matcher:** IDF-weighted keyword overlap with stop word removal. Requires ≥2 keyword matches. Confidence: 50-84%.

**Response Builder:** Three strategies:
1. Template + Data → extracts named groups, looks up data dict, renders template
2. Conditional Answers → scans match/input for condition keywords, returns variant
3. Static Answer → returns the hardcoded answer text

---

## 🚀 How to Run

### Option 1: Jupyter Notebook (recommended for evaluation)

1. Open `regex_chatbot.ipynb` in **Jupyter Notebook / JupyterLab / VS Code / Google Colab**
2. Click **"Run All"** (or run cells one by one)
3. The notebook includes:
   - Problem statement & architecture
   - Regex pattern analysis with feature detection
   - Accuracy testing (sample queries vs. parent patterns)
   - Cross-match testing (correct entry routing)
   - Dynamic response demos (templates, conditionals, dept-specific)
   - Multilingual demo (English, Malayalam, Tamil, Kannada)
   - Conversation flow demo (context tracking, follow-ups)
   - Full test suite
   - Interactive chat with confidence scores

**No extra installations needed!** Uses only built-in Python libraries.

> **Google Colab:** The notebook auto-detects Colab and prompts for file uploads.

### Option 2: Streamlit Web App

```bash
pip install streamlit
streamlit run chatbot.py
```

**Streamlit features:**
- 💬 Modern chat UI with confidence badges (🟢🟡🔴)
- 🏷️ Intent classification tags
- 🌐 Language detection badges
- 💡 Follow-up suggestion buttons
- 📋 Sidebar with categorized topic list
- 🗑️ Clear Chat button

---

## 🔧 Q&A Data Format

Each JSON file contains entries with:

```json
{
  "id": 3,
  "question": "What are the fees for BCA?",
  "category": "ACADEMICS",
  "related_ids": [1, 2],
  "sample_queries": ["BCA fees", "How much does BCA cost?"],
  "pattern": "(?P<programme>bba|bca|mca).*(?:fees?|cost|duration)",
  "answer": "Static fallback answer.",
  "answer_template": "The {programme} has a fee of {fee} for {duration}.",
  "data": {
    "BCA": {"programme": "BCA", "fee": "Rs. 55,485", "duration": "3 years"}
  },
  "conditional_answers": {
    "boy|boys|male": "Boys-specific answer.",
    "girl|girls|female": "Girls-specific answer."
  }
}
```

Only `id`, `question`, `pattern`, and `answer` are required. All other fields are optional enhancements.

---

## 👥 Team Members

| Name       | File        | IDs    | Languages          |
|------------|-------------|--------|-------------------|
| Defitha    | `qa_1.json` | D1–D15 | English, Tamil     |
| Evengiline | `qa_2.json` | E1–E15 | English, Kannada   |
| Sanal      | `qa_3.json` | 1–17   | English, Malayalam |
| Nasreen    | `qa_4.json` | N1–N15 | English, Hindi     |
