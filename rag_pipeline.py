import os
import numpy as np
import fitz  # PyMuPDF
from google import genai
from dotenv import load_dotenv

load_dotenv('config.env')
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Set it in the environment or in .env before running the app."
    )

client = genai.Client(api_key=api_key)


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
    Uses Google Gemini gemini-embedding-001.
    """
    if not texts:
        raise ValueError("No text chunks available for embedding. The PDF may contain only images or no readable text.")

    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=texts
    )
    return [embedding.values for embedding in response.embeddings]

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
    Send retrieved chunks + question + conversation history to Gemini.

    HOW MEMORY WORKS:
    We build a full prompt including:
      1. System instructions
      2. Document context
      3. Conversation history (last 6 messages)
      4. Current question

    Gemini processes the entire context for coherent responses.
    """

    # Build the context block from retrieved chunks
    context = "\n\n---\n\n".join([chunk for chunk, _ in context_chunks])

    # System prompt — sets behaviour for the whole conversation
    system_prompt = """You are a helpful assistant that uses the provided document excerpts
as the primary source of truth.

Rules:
- Answer from the document whenever a direct answer exists.
- If the user asks for brainstorming, related ideas, or explanations
  inspired by the document, you may provide a thoughtful response
  that extends beyond verbatim document content.
- Do not fabricate details about the document itself.
- If no direct answer exists, say:
  "I couldn't find a direct answer in the document, but here is a related idea based on it."
- Be concise, clear, and helpful.
- Use conversation history to interpret follow-up questions."""

    # Build the full prompt
    prompt = f"""{system_prompt}

Here are the relevant excerpts from the document for our conversation:

{context}

Please use these excerpts as the main foundation for your answers.
If the user is asking for brainstorming or related ideas, you may provide relevant
insights based on the document and general topic knowledge."""

    # Add conversation history (last 6 messages)
    recent_history = chat_history[-6:]
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"\n\n{role}: {msg['content']}"

    # Add current question
    prompt += f"\n\nUser: {question}\n\nAssistant:"

    # Generate response with Gemini
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.candidates[0].content.parts[0].text