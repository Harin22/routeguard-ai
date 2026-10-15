from src.rag.vector_store import search_news
from src.rag.geopolitical_analyzer import call_gemini, call_mercury


def chat_with_routeguard(question, history=None, top_k=5, route_key=None):
    """Answer a user question using retrieved maritime news."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    results = search_news(question.strip(), top_k=top_k, route_key=route_key)

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []

    sources = []
    evidence = []
    seen_articles = set()

    for document, metadata in zip(documents, metadatas):
        metadata = metadata or {}

        title = metadata.get("title", "Unknown title")
        source = metadata.get("source", "Unknown source")
        published = metadata.get("published", "Unknown date")
        link = metadata.get("link", "")

        # Remove repeated headlines from legacy ChromaDB records.
        dedup_key = (
            title.strip().lower(),
            source.strip().lower(),
            published.strip().lower(),
        )

        if dedup_key in seen_articles:
            continue

        seen_articles.add(dedup_key)

        evidence.append(
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Published: {published}\n"
            f"URL: {link or 'Unavailable'}\n"
            f"Content: {document}"
        )

        sources.append({
            "title": title,
            "source": source,
            "published": published,
            "url": link or None,
        })

    history_text = ""
    for message in (history or [])[-6:]:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role in {"user", "assistant"} and content:
            history_text += f"{role}: {content}\n"

    evidence_text = "\n\n".join(evidence) or (
        "No relevant articles were retrieved."
    )

    prompt = f"""
You are RouteGuard AI, a maritime risk research assistant.

Answer using the retrieved news evidence.
Do not invent events, dates, sources, or facts.
If evidence is insufficient, say so clearly.
Distinguish reported facts from your interpretation.
Do not claim that an article proves a ship will be disrupted.
Use source titles when referring to evidence.
Mention uncertainty where appropriate.. 

Conversation history:
{history_text or "No previous messages."}

Retrieved news evidence:
{evidence_text}

User question:
{question.strip()}

Provide a clear, concise answer.
"""

    try:
        answer = call_gemini(prompt)
        provider = "Gemini"
    except Exception:
        answer = call_mercury(prompt)
        provider = "Mercury"

    return {
        "answer": answer,
        "sources": sources,
        "provider": provider,
    }
