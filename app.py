"""Streamlit Web Application for the LangChain RAG Chatbot.

Provides a modern, interactive web frontend for document question-answering
and conversation using the LangChain RAG pipeline.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="RAG Chatbot (LangChain)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #4A00E0, #8E2DE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #E8F5E9;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing LangChain RAG Pipeline...")
def get_pipeline(model_name: str = "openai/gpt-oss-20b"):
    """Instantiates and caches the core LocalRAGPipeline."""
    from rag_chatbot.pipeline import LocalRAGPipeline
    return LocalRAGPipeline(host="localhost", default_model=model_name)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = set()

# Initialize Pipeline
pipeline = get_pipeline()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚙️ RAG Settings")

    # API Key status
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key and not groq_key.startswith("your_"):
        st.success("API Key configured", icon="🔑")
    else:
        user_key = st.text_input("Enter Groq API Key:", type="password")
        if user_key:
            os.environ["GROQ_API_KEY"] = user_key
            st.success("API Key updated!", icon="✅")

    # Model Selection
    model_options = [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    ]
    selected_model = st.selectbox(
        "Select LLM Model:",
        options=model_options,
        index=0,
    )
    if selected_model != pipeline.get_model_name():
        pipeline.set_model_name(selected_model)
        pipeline.set_chat_mode()

    # Chat Mode Selection
    chat_mode = st.radio(
        "Conversation Mode:",
        options=["chat", "QA"],
        format_func=lambda x: "Multi-turn Chat (with memory)" if x == "chat" else "Single-turn QA",
        horizontal=True,
    )

    # Language Toggle
    selected_lang = st.selectbox("Language:", options=["English (eng)", "Vietnamese (vi)"], index=0)
    lang_code = "vi" if "vi" in selected_lang else "eng"
    if lang_code != pipeline.get_language():
        pipeline.set_language(lang_code)
        pipeline.set_chat_mode()

    st.markdown("---")
    st.subheader("📄 Document Ingestion")

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Documents:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to index into the RAG vector store.",
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_file_names]
        if new_files:
            with st.spinner(f"Ingesting {len(new_files)} new document(s)..."):
                temp_paths = []
                for uf in new_files:
                    # Save temporarily to parse via PyMuPDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.getbuffer())
                        temp_paths.append(tmp.name)
                        st.session_state.uploaded_file_names.add(uf.name)

                pipeline.store_nodes(temp_paths)
                pipeline.set_engine()
                st.success(f"Indexed {len(new_files)} document(s) successfully!")

    if st.button("🗑️ Reset Ingested Documents", use_container_width=True):
        pipeline.reset_documents()
        pipeline.reset_engine()
        st.session_state.uploaded_file_names.clear()
        st.info("Ingested documents cleared.")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        pipeline.clear_conversation()
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT PANEL ---
st.markdown('<div class="main-header">🤖 Local RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Powered by LangChain & Groq • Hybrid Retrieval (BM25 + Dense) • Neural Cross-Encoder Reranking</div>',
    unsafe_allow_html=True,
)

# Display document status badge
if st.session_state.uploaded_file_names:
    docs_list = ", ".join(st.session_state.uploaded_file_names)
    st.markdown(
        f'<span class="status-badge">📚 Active Documents: {docs_list}</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="status-badge" style="background-color: #FFF3E0; color: #E65100;">💬 General Chat Mode (No documents uploaded)</span>',
        unsafe_allow_html=True,
    )

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    score = getattr(src, "score", 0.0)
                    st.markdown(f"**Chunk [{idx}]** (Score: `{score:.4f}`):")
                    st.text(src.page_content[:300] + ("..." if len(src.page_content) > 300 else ""))

# Handle User Input
if prompt := st.chat_input("Ask a question about your documents or anything else..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format history for pipeline query
    gradio_style_history = [
        [m["content"], None] if m["role"] == "user" else [None, m["content"]]
        for m in st.session_state.messages[:-1]
    ]

    # Query Pipeline and stream output
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Retrieving and generating answer..."):
            stream_response = pipeline.query(
                mode=chat_mode,
                message=prompt,
                chatbot=gradio_style_history,
            )

        # Stream response tokens
        def token_stream():
            for token in stream_response.response_gen:
                yield token

        full_response = response_placeholder.write_stream(token_stream())

        # Show source nodes if documents were retrieved
        sources = getattr(stream_response, "source_nodes", [])
        if sources:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(sources, 1):
                    score = getattr(src, "score", 0.0)
                    st.markdown(f"**Chunk [{idx}]** (Score: `{score:.4f}`):")
                    st.text(src.page_content[:300] + ("..." if len(src.page_content) > 300 else ""))

    # Save assistant message to session state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        }
    )
