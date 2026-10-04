import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")


# --------------------------------------------------
# GEMINI
# --------------------------------------------------

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

GEMINI_MODEL = "gemini-3.8-flash"


# --------------------------------------------------
# CHROMA
# --------------------------------------------------

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "routeguard_news"


# --------------------------------------------------
# COHERE EMBEDDINGS FOR LANGCHAIN
# --------------------------------------------------

class CohereEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper around the Cohere
    embedding model already used by RouteGuard.
    """

    def __init__(self):
        import cohere

        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY not found in .env"
            )

        self.client = cohere.ClientV2(
            api_key=cohere_api_key
        )

    def embed_documents(self, texts):
        response = self.client.embed(
            model="embed-v4.0",
            texts=texts,
            input_type="search_document",
            embedding_types=["float"]
        )

        return response.embeddings.float

    def embed_query(self, text):
        response = self.client.embed(
            model="embed-v4.0",
            texts=[text],
            input_type="search_query",
            embedding_types=["float"]
        )

        return response.embeddings.float[0]


# --------------------------------------------------
# LANGCHAIN VECTOR STORE
# --------------------------------------------------

embeddings = CohereEmbeddings()

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings
)


# --------------------------------------------------
# RETRIEVE NEWS
# --------------------------------------------------

def retrieve_news_for_zone(
    latitude,
    longitude,
    top_k=5
):
    """
    Retrieve the most relevant geopolitical news
    for a route monitoring zone.
    """

    query = (
        "maritime geopolitical risks and "
        "shipping disruptions near "
        f"latitude {latitude:.4f}, "
        f"longitude {longitude:.4f}"
    )

    documents = vector_store.similarity_search(
        query,
        k=top_k
    )

    return documents


# --------------------------------------------------
# GEMINI ANALYSIS
# --------------------------------------------------

def analyze_geopolitical_risk(
    latitude,
    longitude,
    documents
):
    """
    Ask Gemini to analyze retrieved news and
    return one overall geopolitical risk score.
    """

    if not documents:
        return {
            "risk_score": 0,
            "reason": "No relevant geopolitical evidence found."
        }

    evidence = []

    for i, document in enumerate(
        documents,
        start=1
    ):
        evidence.append(
            f"""
SOURCE {i}

{document.page_content}
"""
        )

    evidence_text = "\n".join(evidence)

    prompt = f"""
You are the geopolitical risk analysis engine
for a maritime route risk prediction system.

Monitoring location:
Latitude: {latitude:.4f}
Longitude: {longitude:.4f}

Below are retrieved news documents relevant to
this maritime region.

Analyze the overall geopolitical situation
that could affect maritime shipping.

Return ONLY valid JSON in this exact format:

{{
    "risk_score": 0,
    "reason": "short explanation"
}}

Risk score:
0 = no meaningful geopolitical shipping risk
10 = extremely severe geopolitical shipping risk

Consider:
- armed conflict
- attacks on ships
- blockades
- piracy
- political instability
- sanctions
- major shipping disruptions

Do NOT treat ordinary political news as high risk
unless it has a meaningful connection to maritime
shipping or regional disruption.

Retrieved evidence:

{evidence_text}
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    text = response.text.strip()

    # Remove markdown code fences if Gemini adds them.
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    result = json.loads(text)

    risk_score = float(result["risk_score"])

    # Keep the score within our intended range.
    risk_score = max(
        0,
        min(10, risk_score)
    )

    return {
        "risk_score": risk_score,
        "reason": result.get(
            "reason",
            ""
        )
    }


# --------------------------------------------------
# COMPLETE ZONE ANALYSIS
# --------------------------------------------------

def analyze_zone(
    latitude,
    longitude,
    top_k=5
):
    """
    Complete RAG geopolitical analysis for one zone.
    """

    print(
        f"\nAnalyzing zone: "
        f"{latitude:.4f}, {longitude:.4f}"
    )

    print("Retrieving relevant news...")

    documents = retrieve_news_for_zone(
        latitude,
        longitude,
        top_k=top_k
    )

    print(
        f"Retrieved {len(documents)} documents."
    )

    print("Sending evidence to Gemini...")

    result = analyze_geopolitical_risk(
        latitude,
        longitude,
        documents
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "risk_score": result["risk_score"],
        "reason": result["reason"],
        "sources": [
            document.page_content
            for document in documents
        ]
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    # Example: Red Sea region
    latitude = 20.0
    longitude = 38.0

    result = analyze_zone(
        latitude,
        longitude,
        top_k=5
    )

    print("\n================================")
    print(" GEOPOLITICAL RISK RESULT")
    print("================================")

    print(
        f"\nRisk score: "
        f"{result['risk_score']}/10"
    )

    print(
        f"\nReason:\n"
        f"{result['reason']}"
    )