import io
import os
import re
from typing import List, Dict, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document

# ---------------------------------------------------------
# NoteGenAI: Intelligent Note Generation System
# Compatible with:
# streamlit==1.40.1
# python-dotenv==1.0.1
# requests==2.32.3
# pypdf==5.1.0
# python-docx==1.1.2
#
# AI backend: Ollama (local)
# Default Ollama URL: http://localhost:11434
# ---------------------------------------------------------

load_dotenv()

# ------------------------- CONFIG ------------------------

APP_TITLE = "NoteGenAI"
APP_SUBTITLE = "Intelligent Note Generation System"

DEFAULT_OLLAMA_URL = os.getenv(
    "OLLAMA_URL", "http://localhost:11434"
).rstrip("/")

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

REQUEST_TIMEOUT = 180
CHUNK_SIZE = 7000
CHUNK_OVERLAP = 500

# --------------------- PAGE SETTINGS ----------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- CSS ----------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .info-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 1rem;
        }

        .success-box {
            padding: 0.9rem 1rem;
            border-radius: 10px;
            background: rgba(46, 160, 67, 0.10);
            border: 1px solid rgba(46, 160, 67, 0.35);
        }

        .note-box {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            background: rgba(128,128,128,0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- FUNCTIONS -------------------------

def extract_pdf_text(file_bytes: bytes) -> Tuple[str, int]:
    """Extract text from a PDF and return text + page count."""
    reader = PdfReader(io.BytesIO(file_bytes))

    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append(text.strip())

    full_text = "\n\n".join(
        f"[Page {i + 1}]\n{page_text}"
        for i, page_text in enumerate(pages)
        if page_text
    )

    return full_text.strip(), len(reader.pages)


def extract_docx_text(file_bytes: bytes) -> Tuple[str, int]:
    """Extract paragraphs and tables from a DOCX file."""
    document = Document(io.BytesIO(file_bytes))
    parts = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            row_text = " | ".join(cell for cell in cells if cell)
            if row_text:
                parts.append(row_text)

    text = "\n".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip(), len(document.paragraphs)


def extract_text(uploaded_file) -> Tuple[str, str]:
    """Extract text according to the uploaded file type."""
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        text, count = extract_pdf_text(file_bytes)
        return text, f"{count} PDF page(s)"

    if file_name.endswith(".docx"):
        text, count = extract_docx_text(file_bytes)
        return text, f"{count} DOCX paragraph(s)"

    raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")


def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split long documents into overlapping chunks."""
    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        # Prefer ending at a paragraph/sentence boundary.
        if end < text_length:
            boundary_candidates = [
                chunk.rfind("\n\n"),
                chunk.rfind(". "),
                chunk.rfind(" "),
            ]
            best_boundary = max(boundary_candidates)

            if best_boundary > chunk_size * 0.60:
                end = start + best_boundary

        clean_chunk = text[start:end].strip()
        if clean_chunk:
            chunks.append(clean_chunk)

        next_start = end - overlap
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def ollama_request(
    ollama_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> str:
    """Send a chat request to Ollama."""
    endpoint = f"{ollama_url}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    response = requests.post(
        endpoint,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()
    data = response.json()

    message = data.get("message", {})
    content = message.get("content", "")

    if not content:
        raise RuntimeError("Ollama returned an empty response.")

    return content.strip()


def get_ollama_models(ollama_url: str) -> List[str]:
    """Return locally available Ollama models."""
    try:
        response = requests.get(
            f"{ollama_url}/api/tags",
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        return [
            item.get("name")
            for item in models
            if item.get("name")
        ]
    except Exception:
        return []


def check_ollama(ollama_url: str) -> bool:
    """Check whether the Ollama service is reachable."""
    try:
        response = requests.get(
            f"{ollama_url}/api/tags",
            timeout=5,
        )
        return response.ok
    except requests.RequestException:
        return False


def clean_markdown(text: str) -> str:
    """Light cleanup for generated Markdown."""
    text = text.strip()
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def generate_notes(
    text: str,
    model: str,
    ollama_url: str,
    style: str,
    detail_level: str,
    progress_callback=None,
) -> str:
    """Generate structured notes from a document."""
    chunks = create_chunks(text)

    if not chunks:
        raise ValueError("No readable text was found in the document.")

    detail_instruction = {
        "Short": "Keep the notes concise and focus only on the most important information.",
        "Balanced": "Create clear, useful notes with enough explanation for exam preparation.",
        "Detailed": "Create detailed study notes while avoiding unnecessary repetition.",
    }[detail_level]

    style_instruction = {
        "Exam Notes": (
            "Focus on definitions, key concepts, important facts, formulas, "
            "examples, advantages/disadvantages, and likely exam points."
        ),
        "Easy Revision": (
            "Use simple language, short bullets, headings, keywords, and quick-revision points."
        ),
        "Detailed Study Notes": (
            "Explain concepts clearly with structured headings, subheadings, "
            "examples, and important relationships."
        ),
    }[style]

    partial_notes = []

    for index, chunk in enumerate(chunks, start=1):
        prompt = f"""
You are NoteGenAI, an intelligent academic note-generation assistant.

Create high-quality study notes from the document section below.

Requirements:
- Preserve important facts and meanings.
- Do not invent information that is not supported by the source.
- Use Markdown headings and bullet points.
- Highlight important terms using **bold**.
- Include definitions, concepts, examples, formulas, steps, and comparisons when present.
- Remove repetition and unnecessary filler.
- {style_instruction}
- {detail_instruction}

Document section {index} of {len(chunks)}:
--------------------
{chunk}
--------------------

Generate notes for this section only.
"""

        notes = ollama_request(
            ollama_url=ollama_url,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise educational note-making assistant. "
                        "Use only the supplied source content."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        partial_notes.append(notes)

        if progress_callback:
            progress_callback(index / len(chunks))

    # If the document is short, the section notes are already sufficient.
    if len(partial_notes) == 1:
        return clean_markdown(partial_notes[0])

    combined = "\n\n".join(
        f"### Section {i}\n{notes}"
        for i, notes in enumerate(partial_notes, start=1)
    )

    # Keep the final synthesis request reasonably sized.
    synthesis_chunks = create_chunks(combined, chunk_size=12000, overlap=300)
    synthesis_source = "\n\n".join(synthesis_chunks[:3])

    final_prompt = f"""
You are NoteGenAI. Combine the extracted section notes below into one
well-organized set of final study notes.

Rules:
- Remove duplicated points.
- Preserve important information.
- Use a logical order.
- Start with a clear title.
- Use headings and bullet points.
- Include a "Key Takeaways" section.
- Include "Important Terms" when useful.
- Do not add facts that are not present in the supplied notes.
- Make the final result easy for a student to revise.

Section notes:
--------------------
{synthesis_source}
--------------------

Return only the final Markdown notes.
"""

    final_notes = ollama_request(
        ollama_url=ollama_url,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert academic editor and summarizer.",
            },
            {"role": "user", "content": final_prompt},
        ],
        temperature=0.2,
    )

    return clean_markdown(final_notes)



def generate_questions(text: str, model: str, ollama_url: str, count: int) -> str:
    prompt = f"""
Create {count} exam-oriented questions from the document below.
Use only the supplied document. Include a mixture of short-answer,
conceptual and descriptive questions. Number them clearly.

DOCUMENT:
{text[:18000]}
"""
    return ollama_request(ollama_url, model, [
        {"role": "system", "content": "You are an academic question generator. Do not invent facts."},
        {"role": "user", "content": prompt},
    ], temperature=0.2)


def generate_quiz(text: str, model: str, ollama_url: str, count: int) -> str:
    prompt = f"""
Create a {count}-question multiple-choice quiz from the document.
For each question provide A, B, C, D, the correct answer and a one-line explanation.
Use only information from the document.

DOCUMENT:
{text[:18000]}
"""
    return ollama_request(ollama_url, model, [
        {"role": "system", "content": "You are a precise educational quiz generator."},
        {"role": "user", "content": prompt},
    ], temperature=0.2)


def generate_keywords(text: str, model: str, ollama_url: str) -> str:
    prompt = f"""
Extract 15 to 25 important keywords or technical terms from this document.
For each term, give a short meaning based only on the document.

DOCUMENT:
{text[:15000]}
"""
    return ollama_request(ollama_url, model, [
        {"role": "system", "content": "You extract important academic terms accurately."},
        {"role": "user", "content": prompt},
    ], temperature=0.2)


def answer_document(question: str, text: str, model: str, ollama_url: str) -> str:
    prompt = f"""
Answer the student's question using ONLY the uploaded document.
If the answer is not available, say: The answer is not available in the uploaded document.
Do not invent information.

QUESTION:
{question}

DOCUMENT:
{text[:30000]}
"""
    return ollama_request(ollama_url, model, [
        {"role": "system", "content": "You are a document-grounded educational assistant."},
        {"role": "user", "content": prompt},
    ], temperature=0.1)


def generate_summary(text: str, model: str, ollama_url: str) -> str:
    prompt = f"""
Create a concise study summary of the document below.
Use only the supplied document. Include:
1. Main idea
2. 5-10 key points
3. Final takeaway
Use simple student-friendly language.

DOCUMENT:
{text[:30000]}
"""
    return ollama_request(ollama_url, model, [
        {"role": "system", "content": "You are a precise academic summarizer. Do not invent facts."},
        {"role": "user", "content": prompt},
    ], temperature=0.2)


def generate_flashcards(text: str, model: str, ollama_url: str, count: int) -> str:
    prompt = f"""
Create {count} study flashcards from the document.
Format each as:
### Flashcard 1
**Question:** ...
**Answer:** ...

Use only information from the document. Make questions useful for revision.

DOCUMENT:
{text[:25000]}
"""
    return ollama_request(ollama_url, model, [
        {"role": "system", "content": "You create accurate educational flashcards from source material."},
        {"role": "user", "content": prompt},
    ], temperature=0.2)


def generate_topics(text: str, model: str, ollama_url: str) -> str:
    prompt = f"""
Identify the most important topics in this document.
For each topic give:
- Topic name
- Priority: High, Medium, or Low
- Why it is important
- What a student should remember

Use only the document. Present the result as a clear Markdown table.

DOCUMENT:
{text[:30000]}
"""
    return ollama_request(ollama_url, model, [
        {"role": "system", "content": "You are an academic topic-priority analyzer. Do not invent information."},
        {"role": "user", "content": prompt},
    ], temperature=0.2)


def analyze_difficulty(text: str, model: str, ollama_url: str) -> str:
    prompt = f"""
Analyze the study material and classify its major topics as Easy, Medium, or Difficult.
For each topic provide:
- Topic
- Difficulty
- Reason
- Suggested study action

Then provide a short overall difficulty assessment.
Use only information from the document. Do not judge the student's personal ability.

DOCUMENT:
{text[:30000]}
"""
    return ollama_request(ollama_url, model, [
        {"role": "system", "content": "You are an educational difficulty analyzer. Be neutral and source-grounded."},
        {"role": "user", "content": prompt},
    ], temperature=0.2)

# ------------------------- SIDEBAR ------------------------

with st.sidebar:
    st.header("⚙️ NoteGenAI Settings")

    ollama_url = st.text_input(
        "Ollama URL",
        value=DEFAULT_OLLAMA_URL,
        help="Default: http://localhost:11434",
    ).rstrip("/")

    available_models = get_ollama_models(ollama_url)

    if available_models:
        default_index = (
            available_models.index(DEFAULT_MODEL)
            if DEFAULT_MODEL in available_models
            else 0
        )

        model = st.selectbox(
            "AI Model",
            available_models,
            index=default_index,
        )
        st.success("Ollama is connected.")
    else:
        model = st.text_input(
            "AI Model",
            value=DEFAULT_MODEL,
            help="Enter the Ollama model name installed on your computer.",
        )
        st.warning(
            "No Ollama models detected. Make sure Ollama is running "
            "and a model is installed."
        )

    st.divider()

    note_style = st.selectbox(
        "Note style",
        [
            "Exam Notes",
            "Easy Revision",
            "Detailed Study Notes",
        ],
    )

    detail_level = st.select_slider(
        "Detail level",
        options=["Short", "Balanced", "Detailed"],
        value="Balanced",
    )

    st.divider()

    st.caption("Supported files")
    st.write("📄 PDF")
    st.write("📝 DOCX")

# -------------------------- HEADER ------------------------

st.markdown(
    '<div class="main-title">📝 NoteGenAI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Intelligent Note Generation System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
        <b>What does NoteGenAI do?</b><br>
        Upload a PDF or Word document and let AI transform lengthy study
        material into clear, structured and revision-friendly notes.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------- STATUS ----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("AI Backend", "Ollama")

with col2:
    st.metric("Input Formats", "PDF / DOCX")

with col3:
    status = "Online" if check_ollama(ollama_url) else "Offline"
    st.metric("Ollama Status", status)

# ---------------------- FILE UPLOAD -----------------------

uploaded_file = st.file_uploader(
    "📂 Upload your study material",
    type=["pdf", "docx"],
    help="Upload a PDF or Microsoft Word document.",
)

if uploaded_file:
    st.success(f"Uploaded: **{uploaded_file.name}**")

    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""

    if "generated_notes" not in st.session_state:
        st.session_state.generated_notes = ""

    if "generated_questions" not in st.session_state:
        st.session_state.generated_questions = ""

    if "generated_quiz" not in st.session_state:
        st.session_state.generated_quiz = ""

    if "generated_keywords" not in st.session_state:
        st.session_state.generated_keywords = ""

    if "document_answer" not in st.session_state:
        st.session_state.document_answer = ""

    if "generated_summary" not in st.session_state:
        st.session_state.generated_summary = ""

    if "generated_flashcards" not in st.session_state:
        st.session_state.generated_flashcards = ""

    if "generated_topics" not in st.session_state:
        st.session_state.generated_topics = ""

    if "difficulty_analysis" not in st.session_state:
        st.session_state.difficulty_analysis = ""

    # ---------- Extract text ----------
    with st.expander("🔍 Preview extracted text", expanded=False):
        try:
            extracted_text, document_info = extract_text(uploaded_file)

            st.info(
                f"Document information: **{document_info}** | "
                f"Extracted characters: **{len(extracted_text):,}**"
            )

            if extracted_text:
                st.text_area(
                    "Extracted text",
                    extracted_text[:10000],
                    height=300,
                )
            else:
                st.warning(
                    "No readable text was found. If this is a scanned PDF, "
                    "OCR may be required."
                )

            st.session_state.extracted_text = extracted_text

        except Exception as error:
            st.error(f"Could not read the document: {error}")
            st.stop()

    # ---------- Generate notes ----------
    st.divider()

    generate_button = st.button(
        "✨ Generate Intelligent Notes",
        type="primary",
        use_container_width=True,
    )

    if generate_button:
        if not st.session_state.extracted_text.strip():
            st.error("No text was extracted from the uploaded document.")
            st.stop()

        if not check_ollama(ollama_url):
            st.error(
                "Ollama is not running or cannot be reached. "
                "Start Ollama and try again."
            )
            st.stop()

        progress_bar = st.progress(0)
        status_box = st.empty()

        def update_progress(value):
            progress_bar.progress(min(int(value * 100), 100))
            status_box.info(
                f"Generating notes... {int(value * 100)}% complete"
            )

        try:
            with st.spinner("AI is generating your notes..."):
                notes = generate_notes(
                    text=st.session_state.extracted_text,
                    model=model,
                    ollama_url=ollama_url,
                    style=note_style,
                    detail_level=detail_level,
                    progress_callback=update_progress,
                )

            st.session_state.generated_notes = notes
            progress_bar.progress(100)
            status_box.success("Notes generated successfully! 🎉")

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to Ollama. Please make sure Ollama "
                "is running on your computer."
            )
        except requests.exceptions.HTTPError as error:
            st.error(f"Ollama returned an HTTP error: {error}")
        except Exception as error:
            st.error(f"Note generation failed: {error}")

# ---------------------- EXTRA AI FEATURES ------------------

if st.session_state.get("extracted_text"):
    st.divider()
    st.subheader("🚀 More NoteGenAI Features")
    st.caption("Turn the same document into summaries, questions, quizzes, flashcards, keywords and personalized study guidance.")

    feature_tabs = st.tabs([
        "📌 Summary",
        "❓ Questions",
        "🧠 AI Quiz",
        "🗂️ Flashcards",
        "🔑 Keywords",
        "🎯 Important Topics",
        "📊 Difficulty",
        "💬 Ask Document",
    ])

    # Summary
    with feature_tabs[0]:
        st.write("Create a quick revision summary from the uploaded document.")
        if st.button("Generate Study Summary", key="summary_btn", type="primary", use_container_width=True):
            if not check_ollama(ollama_url):
                st.error("Ollama is not running. Start Ollama and try again.")
            else:
                try:
                    with st.spinner("Creating study summary..."):
                        st.session_state.generated_summary = generate_summary(
                            st.session_state.extracted_text, model, ollama_url
                        )
                    st.success("Summary generated successfully!")
                except Exception as error:
                    st.error(f"Summary generation failed: {error}")
        if st.session_state.generated_summary:
            st.markdown(st.session_state.generated_summary)
            st.download_button(
                "⬇️ Download Summary",
                st.session_state.generated_summary,
                "NoteGenAI_Summary.md",
                "text/markdown",
                use_container_width=True,
            )

    # Question Generator
    with feature_tabs[1]:
        st.write("Generate exam-oriented questions directly from the uploaded document.")
        question_count = st.slider("Number of questions", 5, 20, 10, key="question_count")
        if st.button("Generate Questions", key="generate_questions_btn", use_container_width=True):
            if not check_ollama(ollama_url):
                st.error("Ollama is not running. Start Ollama and try again.")
            else:
                try:
                    with st.spinner("Generating exam questions..."):
                        st.session_state.generated_questions = generate_questions(
                            st.session_state.extracted_text, model, ollama_url, question_count
                        )
                    st.success("Questions generated successfully!")
                except Exception as error:
                    st.error(f"Question generation failed: {error}")
        if st.session_state.generated_questions:
            st.markdown(st.session_state.generated_questions)
            st.download_button("⬇️ Download Questions", st.session_state.generated_questions,
                               "NoteGenAI_Questions.md", "text/markdown", use_container_width=True)

    # AI Quiz
    with feature_tabs[2]:
        st.write("Create multiple-choice questions with answers and explanations.")
        quiz_count = st.slider("Number of MCQs", 5, 15, 10, key="quiz_count")
        if st.button("Generate AI Quiz", key="generate_quiz_btn", use_container_width=True):
            if not check_ollama(ollama_url):
                st.error("Ollama is not running. Start Ollama and try again.")
            else:
                try:
                    with st.spinner("Creating your quiz..."):
                        st.session_state.generated_quiz = generate_quiz(
                            st.session_state.extracted_text, model, ollama_url, quiz_count
                        )
                    st.success("Quiz generated successfully!")
                except Exception as error:
                    st.error(f"Quiz generation failed: {error}")
        if st.session_state.generated_quiz:
            st.markdown(st.session_state.generated_quiz)
            st.download_button("⬇️ Download Quiz", st.session_state.generated_quiz,
                               "NoteGenAI_Quiz.md", "text/markdown", use_container_width=True)

    # Flashcards
    with feature_tabs[3]:
        st.write("Generate question-and-answer flashcards for active recall.")
        flashcard_count = st.slider("Number of flashcards", 5, 20, 10, key="flashcard_count")
        if st.button("Generate Flashcards", key="flashcards_btn", type="primary", use_container_width=True):
            if not check_ollama(ollama_url):
                st.error("Ollama is not running. Start Ollama and try again.")
            else:
                try:
                    with st.spinner("Creating flashcards..."):
                        st.session_state.generated_flashcards = generate_flashcards(
                            st.session_state.extracted_text, model, ollama_url, flashcard_count
                        )
                    st.success("Flashcards generated successfully!")
                except Exception as error:
                    st.error(f"Flashcard generation failed: {error}")
        if st.session_state.generated_flashcards:
            st.markdown(st.session_state.generated_flashcards)
            st.download_button("⬇️ Download Flashcards", st.session_state.generated_flashcards,
                               "NoteGenAI_Flashcards.md", "text/markdown", use_container_width=True)

    # Keywords
    with feature_tabs[4]:
        st.write("Extract important terms and their meanings for quick revision.")
        if st.button("Extract Important Keywords", key="keywords_btn", use_container_width=True):
            if not check_ollama(ollama_url):
                st.error("Ollama is not running. Start Ollama and try again.")
            else:
                try:
                    with st.spinner("Finding important keywords..."):
                        st.session_state.generated_keywords = generate_keywords(
                            st.session_state.extracted_text, model, ollama_url
                        )
                    st.success("Keywords extracted successfully!")
                except Exception as error:
                    st.error(f"Keyword extraction failed: {error}")
        if st.session_state.generated_keywords:
            st.markdown(st.session_state.generated_keywords)

    # Important Topics
    with feature_tabs[5]:
        st.write("Find high-priority topics that deserve more attention during revision.")
        if st.button("Analyze Important Topics", key="topics_btn", type="primary", use_container_width=True):
            if not check_ollama(ollama_url):
                st.error("Ollama is not running. Start Ollama and try again.")
            else:
                try:
                    with st.spinner("Analyzing important topics..."):
                        st.session_state.generated_topics = generate_topics(
                            st.session_state.extracted_text, model, ollama_url
                        )
                    st.success("Important topics identified!")
                except Exception as error:
                    st.error(f"Topic analysis failed: {error}")
        if st.session_state.generated_topics:
            st.markdown(st.session_state.generated_topics)
            st.download_button("⬇️ Download Topic Analysis", st.session_state.generated_topics,
                               "NoteGenAI_Important_Topics.md", "text/markdown", use_container_width=True)

    # Difficulty Analyzer
    with feature_tabs[6]:
        st.write("Estimate topic difficulty from the complexity of the uploaded material.")
        st.info("This estimates the complexity of the material, not your personal ability.")
        if st.button("Analyze Difficulty", key="difficulty_btn", use_container_width=True):
            if not check_ollama(ollama_url):
                st.error("Ollama is not running. Start Ollama and try again.")
            else:
                try:
                    with st.spinner("Analyzing topic difficulty..."):
                        st.session_state.difficulty_analysis = analyze_difficulty(
                            st.session_state.extracted_text, model, ollama_url
                        )
                    st.success("Difficulty analysis completed!")
                except Exception as error:
                    st.error(f"Difficulty analysis failed: {error}")
        if st.session_state.difficulty_analysis:
            st.markdown(st.session_state.difficulty_analysis)
            st.download_button("⬇️ Download Difficulty Analysis", st.session_state.difficulty_analysis,
                               "NoteGenAI_Difficulty_Analysis.md", "text/markdown", use_container_width=True)

    # Ask My Document
    with feature_tabs[7]:
        st.write("Ask questions about the uploaded document and get document-based answers.")
        question = st.text_area(
            "Your question",
            placeholder="Example: Explain the main advantages mentioned in the document.",
            height=110,
            key="document_question",
        )
        if st.button("Ask AI", key="ask_document_btn", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question.")
            elif not check_ollama(ollama_url):
                st.error("Ollama is not running. Start Ollama and try again.")
            else:
                try:
                    with st.spinner("Reading the document and preparing the answer..."):
                        st.session_state.document_answer = answer_document(
                            question, st.session_state.extracted_text, model, ollama_url
                        )
                    st.success("Answer generated!")
                except Exception as error:
                    st.error(f"Could not answer the question: {error}")
        if st.session_state.document_answer:
            st.markdown("### 🤖 AI Answer")
            st.markdown(st.session_state.document_answer)

# ------------------------- FOOTER -------------------------

st.divider()
st.caption(
    "NoteGenAI • Intelligent Note Generation System • "
    "Python + Streamlit + pypdf + python-docx + Ollama"
)
