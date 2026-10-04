import os
from pathlib import Path

import chromadb
import cohere
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError(
        "COHERE_API_KEY not found in .env"
    )

EMBEDDING_MODEL = "embed-v4.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "routeguard_news"

# Cohere allows a maximum of 96 texts per request.
# We use 50 to stay safely below the limit.
BATCH_SIZE = 50


# =========================================================
# CLIENTS
# =========================================================

co = cohere.ClientV2(
    api_key=COHERE_API_KEY
)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def create_embeddings(texts):
    """
    Convert news article text into Cohere embeddings.
    """

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
    """
    Create embeddings for news articles and
    store them in ChromaDB in batches.
    """

    if not articles:
        print("No articles to store.")
        return

    documents = []
    ids = []
    metadatas = []

    # -----------------------------------------------------
    # Prepare documents
    # -----------------------------------------------------

    for article in articles:

        title = article.get(
            "title",
            ""
        )

        description = article.get(
            "description",
            ""
        )

        content = article.get(
            "content",
            ""
        )

        text = (
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Content: {content}"
        )

        documents.append(text)

        article_id = str(
            article.get("id")
            or article.get("article_id")
            or len(ids)
        )

        ids.append(article_id)

        metadatas.append({
            "title": title,

            "source": article.get(
                "source",
                "unknown"
            ),

            "published": article.get(
                "published",
                ""
            ),

            "location": article.get(
                "location",
                ""
            ),

            "zone_id": str(
                article.get(
                    "zone_id",
                    ""
                )
            ),
        })

    # -----------------------------------------------------
    # Process in batches
    # -----------------------------------------------------

    total_articles = len(documents)

    print(
        f"\nTotal articles to embed: "
        f"{total_articles}"
    )

    for start in range(
        0,
        total_articles,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total_articles
        )

        batch_documents = documents[
            start:end
        ]

        batch_ids = ids[
            start:end
        ]

        batch_metadatas = metadatas[
            start:end
        ]

        print(
            f"\nEmbedding articles "
            f"{start + 1}-{end} "
            f"of {total_articles}..."
        )

        # Create embeddings for this batch
        embeddings = create_embeddings(
            batch_documents
        )

        # Store this batch in ChromaDB
        collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=embeddings,
            metadatas=batch_metadatas,
        )

        print(
            f"Stored articles "
            f"{start + 1}-{end} "
            f"in ChromaDB."
        )

    print(
        f"\nSuccessfully stored "
        f"{total_articles} articles "
        f"in ChromaDB."
    )


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def search_news(
    query,
    top_k=5
):
    """
    Search ChromaDB using semantic similarity.
    """

    response = co.embed(
        model=EMBEDDING_MODEL,
        input_type="search_query",
        texts=[query],
        embedding_types=["float"],
    )

    query_embedding = (
        response.embeddings.float[0]
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
    )

    return results


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "\n================================"
    )

    print(
        " ROUTEGUARD VECTOR STORE"
    )

    print(
        "================================"
    )

    print(
        f"\nEmbedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"ChromaDB path: "
        f"{CHROMA_PATH}"
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"\nCurrent documents: "
        f"{collection.count()}"
    )

    print(
        "\nChromaDB initialized successfully."
    )