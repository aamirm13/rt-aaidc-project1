import os
import streamlit as st

# Import RAGAssistant without changing existing files.
try:
    from app import RAGAssistant
except Exception as e1:
    raise ImportError(
        "Could not import RAGAssistant from app.py. "
        "Make sure streamlit_app.py is in the same folder as app.py."
    ) from e1


#  Page config 
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="💬",
    layout="centered",
)


#  Styling simple & clean
st.markdown(
    """
    <style>
      .chat-wrap { max-width: 820px; margin: 0 auto; }
      .bubble {
        padding: 12px 14px;
        border-radius: 18px;
        margin: 8px 0;
        line-height: 1.35;
        display: inline-block;
        max-width: 85%;
        word-wrap: break-word;
        white-space: pre-wrap;
        font-size: 0.98rem;
      }
      .user {
        background: #0b84ff;
        color: white;
        float: right;
        clear: both;
        border-bottom-right-radius: 6px;
      }
      .assistant {
        background: #f1f3f5;
        color: #111;
        float: left;
        clear: both;
        border-bottom-left-radius: 6px;
      }
      .meta {
        font-size: 0.85rem;
        color: #666;
        margin-top: 8px;
      }
      .welcome-card {
        background: #ffffff;
        border: 1px solid #eee;
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
      }
      .subtle {
        color: #555;
      }
      .divider {
        height: 1px;
        background: #eee;
        margin: 18px 0;
      }

      /* Make the notice OK button a small circular sky-blue button */
      div[data-testid="stButton"] > button[data-testid="baseButton-primary"][aria-label="dismiss_start_notice"],
      div[data-testid="stButton"] > button[data-testid="baseButton-secondary"][aria-label="dismiss_start_notice"],
      div[data-testid="stButton"] > button[aria-label="dismiss_start_notice"] {
        width: 34px !important;
        height: 34px !important;
        padding: 0 !important;
        border-radius: 999px !important;
        background: #e8f2ff !important;      /* same shade as notice */
        border: 1px solid #b9d7ff !important;
        color: #0b3d91 !important;
        font-weight: 700 !important;
        min-width: 34px !important;
      }
      div[data-testid="stButton"] > button[aria-label="dismiss_start_notice"]:hover {
        background: #d9ebff !important;
        border-color: #9fc7ff !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


#  App state
if "page" not in st.session_state:
    st.session_state.page = "welcome"  # welcome -> chat
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str, "sources": optional}
if "show_start_notice" not in st.session_state:
    st.session_state.show_start_notice = False


def get_or_create_assistant() -> RAGAssistant:
    """Create the assistant once per session (keeps it fast and consistent)."""
    if st.session_state.assistant is None:
        a = RAGAssistant()

        # Auto-ingest if DB empty
        if a.vector_db.count() == 0:
            data_dir = os.getenv("DATA_DIR", "data")
            a.ingest(data_dir)

        st.session_state.assistant = a
    return st.session_state.assistant


#  Welcome page
def render_welcome():
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    st.title("RAG-Based AI Assistant")
    st.markdown(
        """
        <div class="welcome-card">
          <p class="subtle">
            This assistant uses <b>Retrieval-Augmented Generation (RAG)</b> to answer questions using
            your local document knowledge base.
          </p>
          <div class="divider"></div>
          <p><b>How it works (in simple terms):</b></p>
          <ul>
            <li>Your documents are split into overlapping chunks (to preserve context).</li>
            <li>Each chunk is embedded into a vector space and stored in a persistent ChromaDB database.</li>
            <li>When you ask a question, the assistant retrieves the most relevant chunks.</li>
            <li>It generates an answer using <b>only</b> the retrieved context, with citations.</li>
            <li>If the answer isn’t in the documents, it refuses rather than hallucinating.</li>
          </ul>
          <p class="subtle">
            Tip: Ask questions directly related to the topics in your <code>data/</code> folder.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("Begin", use_container_width=True):
        st.session_state.page = "chat"
        st.session_state.show_start_notice = True
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# Chat page
def render_chat():
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    st.title("Chat")

    if st.session_state.show_start_notice:
        notice_cols = st.columns([12, 1])
        with notice_cols[0]:
            st.markdown(
                """
                <div style="
                    background-color: #e8f2ff;
                    padding: 12px 14px;
                    border-radius: 10px;
                    font-size: 0.95rem;
                    color: #0b3d91;
                ">
                  Please allow 5–10 seconds after sending your first message for the chat to begin.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with notice_cols[1]:
            if st.button("OK", key="dismiss_start_notice", help="Dismiss"):
                st.session_state.show_start_notice = False
                st.rerun()

    # small header
    st.caption("Grounded answers with source citations • Type a question below")

    # Render history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        bubble_class = "user" if role == "user" else "assistant"
        st.markdown(
            f'<div class="bubble {bubble_class}">{content}</div>',
            unsafe_allow_html=True,
        )
        # Clear floats
        st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

        # Show sources under assistant messages if provided
        if role == "assistant" and msg.get("sources"):
            with st.expander("Sources & distance scores"):
                for s in msg["sources"]:
                    st.write(f"- {s['source']} | {s['chunk_id']} | distance={s['distance']:.4f}")

    st.write("")  # spacing

    # New chat button
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Input row (bottom)
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_input("Message", placeholder="Type your question…")
        submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_text.strip():
        assistant = get_or_create_assistant()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_text.strip()})

        # Get answer
        result = assistant.invoke(user_text.strip(), n_results=int(os.getenv("TOP_K", "5")), show_scores=True)

        # Add assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get("answer", ""),
                "sources": result.get("sources", []),
            }
        )
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# Router 
if st.session_state.page == "welcome":
    render_welcome()
else:
    render_chat()
