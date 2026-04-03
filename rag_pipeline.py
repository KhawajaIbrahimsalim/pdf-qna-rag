import os
import numpy as np
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv('config.env')
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Set it in the environment or in .env before running the app."
    )

client = OpenAI(api_key=api_key)


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF, tagging each page."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages)


def chunk_text(text, chunk_size=400, overlap=60):
    """
    Split text into overlapping word-based chunks.
    Overlap ensures context isn't lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def get_embeddings(texts):
    """
    Convert a list of text strings into embedding vectors.
    Uses OpenAI text-embedding-3-small — cheap and accurate.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a, b):
    """Measure how similar two vectors are (1.0 = identical meaning)."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=4):
    """
    Find the k chunks most semantically similar to the query.
    This is the 'retrieval' step in RAG.
    """
    scores = [
        cosine_similarity(query_embedding, emb)
        for emb in chunk_embeddings
    ]
    top_indices = np.argsort(scores)[::-1][:k]
    return [(chunks[i], scores[i]) for i in top_indices]


def answer_question(question, context_chunks, chat_history):
    """
    Send retrieved chunks + question + conversation history to GPT-4o-mini.

    HOW MEMORY WORKS:
    Instead of sending a single message, we build a full messages array:
      1. A system message setting the assistant's behaviour and rules
      2. The document context injected once as a user message
      3. The last N turns of real conversation (alternating user/assistant)
      4. The current question as the final user message

    GPT sees the whole conversation so follow-ups like
    "can you elaborate?" or "what about point 2?" work correctly.
    We cap history at 6 turns (3 exchanges) to control token costs.
    """

    # Build the context block from retrieved chunks
    context = "\n\n---\n\n".join([chunk for chunk, _ in context_chunks])

    # System prompt — sets behaviour for the whole conversation
    system_prompt = """You are a helpful assistant that answers questions 
based strictly on document excerpts provided to you.

Rules:
- Only use information from the document context. Do not make things up.
- If the answer is not in the context, say:
  "I couldn't find this in the document."
- Be concise and specific.
- When answering follow-up questions, use the conversation history
  to understand what the user is referring to."""

    # Start building the messages array
    messages = [{"role": "system", "content": system_prompt}]

    # Inject document context as the first user message.
    # We frame it as a setup message so it stays in memory
    # across all turns without re-sending every time.
    messages.append({
        "role": "user",
        "content": f"""Here are the relevant excerpts from the document 
for our conversation:

{context}

Please use these excerpts to answer my questions."""
    })

    # Acknowledge context injection (keeps message pairs balanced)
    messages.append({
        "role": "assistant",
        "content": "Understood. I have read the document excerpts. "
                   "Please go ahead with your questions."
    })

    # Add the last 6 messages from real conversation history
    # (= last 3 user questions + 3 assistant answers)
    # Capped to avoid inflating token costs on long sessions
    recent_history = chat_history[-6:]
    for msg in recent_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Finally, add the current question
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        max_tokens=600
    )
    return response.choices[0].message.content