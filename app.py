import streamlit as st
from rag_pipeline import (
    extract_text_from_pdf,
    chunk_text,
    get_embeddings,
    retrieve_top_k,
    answer_question
)

st.set_page_config(
    page_title="PDF QnA Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF QnA Assistant")
st.caption("Upload any PDF and ask questions about it.")


# --- Session state setup ---
# Streamlit reruns the entire script on every interaction.
# session_state persists data across reruns so we don't
# re-index the PDF or lose the chat on every keypress.
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "chat_history" not in st.session_state:
    # Stores the clean conversation: [{"role": "user"/"assistant", "content": "..."}]
    # This is what gets passed into answer_question for memory
    st.session_state.chat_history = []
if "display_history" not in st.session_state:
    # Stores richer display data including source excerpts
    # Kept separate so memory passed to LLM stays clean text only
    st.session_state.display_history = []
if "filename" not in st.session_state:
    st.session_state.filename = None


# --- Sidebar: PDF upload and indexing ---
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        if uploaded_file.name != st.session_state.filename:
            if st.button("Index Document", type="primary"):
                with st.spinner("Reading and indexing..."):
                    text = extract_text_from_pdf(uploaded_file)
                    chunks = chunk_text(text)
                    if not chunks:
                        st.error("No readable text was found in this PDF. Try a text-based PDF instead.")
                    else:
                        embeddings = get_embeddings(chunks)

                        st.session_state.chunks = chunks
                        st.session_state.embeddings = embeddings
                        st.session_state.chat_history = []
                        st.session_state.display_history = []
                        st.session_state.filename = uploaded_file.name

                st.success(f"Done! {len(chunks)} chunks indexed.")

    if st.session_state.filename:
        st.info(f"Loaded: {st.session_state.filename}")
        st.caption(f"{len(st.session_state.chunks)} chunks in memory")
        st.caption(
            f"{len(st.session_state.chat_history) // 2} "
            f"exchanges in memory"
        )

        if st.button("Clear & reset"):
            for key in [
                "chunks", "embeddings",
                "chat_history", "display_history", "filename"
            ]:
                st.session_state[key] = None
            st.rerun()

    # Memory usage indicator
    if st.session_state.chat_history:
        turns_in_memory = min(len(st.session_state.chat_history), 6)
        st.divider()
        st.caption(
            f"Conversation memory: last "
            f"{turns_in_memory} messages active"
        )


# --- Main area: chat interface ---
if st.session_state.chunks is None:
    st.info("Upload a PDF in the sidebar to get started.")

else:
    # Render existing display history (includes source excerpts)
    for msg in st.session_state.display_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("View source excerpts"):
                    for i, (chunk, score) in enumerate(msg["sources"]):
                        st.markdown(
                            f"**Excerpt {i+1}** — "
                            f"relevance score: `{score:.2f}`"
                        )
                        st.text(
                            chunk[:400] + "..."
                            if len(chunk) > 400 else chunk
                        )

    # Question input
    question = st.chat_input("Ask anything about the document...")

    if question:
        # Render user message immediately
        with st.chat_message("user"):
            st.write(question)

        # Save to both history stores
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        st.session_state.display_history.append(
            {"role": "user", "content": question}
        )

        # Retrieve relevant chunks and generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):

                # Embed the question
                query_emb = get_embeddings([question])[0]

                # Find most relevant chunks
                top_chunks = retrieve_top_k(
                    query_emb,
                    st.session_state.embeddings,
                    st.session_state.chunks,
                    k=4
                )

                # Generate answer with full conversation memory.
                # We pass chat_history EXCLUDING the current question
                # because answer_question appends it separately.
                history_so_far = st.session_state.chat_history[:-1]
                answer = answer_question(
                    question,
                    top_chunks,
                    history_so_far
                )

            st.write(answer)

            with st.expander("View source excerpts"):
                for i, (chunk, score) in enumerate(top_chunks):
                    st.markdown(
                        f"**Excerpt {i+1}** — "
                        f"relevance score: `{score:.2f}`"
                    )
                    st.text(
                        chunk[:400] + "..."
                        if len(chunk) > 400 else chunk
                    )

        # Save assistant response to both history stores
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        st.session_state.display_history.append({
            "role": "assistant",
            "content": answer,
            "sources": top_chunks
        })