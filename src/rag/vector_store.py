
import os
import hashlib
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import chromadb
import cohere
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env")

EMBEDDING_MODEL = "embed-v4.0"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "routeguard_news"
BATCH_SIZE = 50


# =========================================================
# CLIENTS
# =========================================================

co = cohere.ClientV2(api_key=COHERE_API_KEY)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================================================
# HELPERS
# =========================================================

def normalize_url(url):
    """Normalize a URL for use as a fallback article identity."""
    if not url:
        return ""

    parts = urlsplit(url.strip())

    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        parts.path.rstrip("/"),
        parts.query,
        "",
    ))


def stable_article_id(article):
    """Prefer the provider's ID, then the canonical URL."""
    article_id = article.get("article_id")

    if article_id:
        identity = f"newsdata:{str(article_id).strip()}"
    else:
        url = normalize_url(article.get("link", ""))

        if not url:
            raise ValueError(
                "Article requires article_id or link."
            )

        identity = f"url:{url}"

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def create_embeddings(texts):
    """Embed article documents using Cohere."""
    response = co.embed(
        model=EMBEDDING_MODEL,
        input_type="search_document",
        texts=texts,
        embedding_types=["float"],
    )

    return response.embeddings.float


# =========================================================
# STORE NEWS ARTICLES
# =========================================================

def add_news_articles(articles):
    """Deduplicate and upsert news articles into ChromaDB."""

    if not articles:
        print("No articles to store.")
        return

    # Merge repeated article IDs within this ingestion batch.
    unique_articles = {}

    for article in articles:
        try:
            doc_id = stable_article_id(article)
        except ValueError:
            print("Skipping article without an ID or URL.")
            continue

        if doc_id not in unique_articles:
            unique_articles[doc_id] = {
                **article,
                "zone_ids": list(
                    dict.fromkeys(
                        str(z)
                        for z in article.get(
                            "zone_ids",
                            [article.get("zone_id", "")],
                        )
                        if str(z)
                    )
                ),
            }
        else:
            existing = unique_articles[doc_id]

            incoming_zones = article.get(
                "zone_ids",
                [article.get("zone_id", "")],
            )

            existing["zone_ids"] = list(dict.fromkeys(
                existing["zone_ids"]
                + [str(z) for z in incoming_zones if str(z)]
            ))

            # Retain useful fields when the first result is incomplete.
            for field in (
                "title",
                "description",
                "content",
                "source",
                "published",
                "link",
            ):
                if not existing.get(field) and article.get(field):
                    existing[field] = article[field]

    records = []

    for doc_id, article in unique_articles.items():
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")

        document = (
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Content: {content}"
        )

        zone_ids = article.get("zone_ids", [])

        metadata = {
            "title": title or "Unknown",
            "source": article.get("source", "unknown") or "unknown",
            "published": article.get("published", "") or "",
            "location": article.get("location", "") or "",
            "zone_id": zone_ids[0] if zone_ids else "",
            "zone_ids": ",".join(zone_ids),
            "article_id": str(article.get("article_id", "") or ""),
            "link": str(article.get("link", "") or ""),
        }

        records.append((doc_id, document, metadata))

    print(f"\nUnique articles to store: {len(records)}")

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        ids = [record[0] for record in batch]
        documents = [record[1] for record in batch]
        metadatas = [record[2] for record in batch]

        print(
            f"Embedding articles {start + 1}-"
            f"{start + len(batch)} of {len(records)}..."
        )

        embeddings = create_embeddings(documents)

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"Successfully upserted {len(records)} unique articles.")


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def search_news(query, top_k=5):
    """Retrieve relevant news documents using semantic similarity."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if collection.count() == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }

    response = co.embed(
        model=EMBEDDING_MODEL,
        input_type="search_query",
        texts=[query.strip()],
        embedding_types=["float"],
    )

    query_embedding = response.embeddings.float[0]

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    print("ROUTEGUARD VECTOR STORE")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"ChromaDB path: {CHROMA_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Current documents: {collection.count()}")
    print("ChromaDB initialized successfully.")
