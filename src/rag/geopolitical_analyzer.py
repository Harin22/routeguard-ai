import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

GEMINI_MODEL = "gemini-3.8-flash"

from src.data.route_sampler import generate_maritime_route
from src.data.risk_zone import create_risk_zones


CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "routeguard_news"


class CohereEmbeddings(Embeddings):

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


embeddings = CohereEmbeddings()

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings
)


def retrieve_news_for_zone(
    latitude,
    longitude,
    top_k=5
):

    query = (
        "maritime geopolitical risks and "
        "shipping disruptions near "
        f"latitude {latitude:.4f}, "
        f"longitude {longitude:.4f}"
    )

    return vector_store.similarity_search(
        query,
        k=top_k
    )


def collect_route_evidence(
    zones,
    top_k=5
):

    evidence = []

    for zone in zones:

        print(
            f"\nZone {zone['zone_id']:02d}"
        )

        print(
            f"Location: "
            f"{zone['latitude']:.4f}, "
            f"{zone['longitude']:.4f}"
        )

        print("Retrieving news...")

        documents = retrieve_news_for_zone(
            zone["latitude"],
            zone["longitude"],
            top_k=top_k
        )

        print(
            f"Retrieved {len(documents)} documents."
        )

        zone_evidence = []

        for document in documents:

            zone_evidence.append(
                document.page_content
            )

        evidence.append({
            "zone_id": zone["zone_id"],
            "latitude": zone["latitude"],
            "longitude": zone["longitude"],
            "days_from_origin": zone[
                "days_from_origin"
            ],
            "documents": zone_evidence
        })

    return evidence


def analyze_route_geopolitical_risk(
    route,
    evidence
):

    evidence_text = []

    for zone in evidence:

        evidence_text.append(
            f"""
ZONE {zone['zone_id']}

Location:
Latitude: {zone['latitude']:.4f}
Longitude: {zone['longitude']:.4f}

Approximate position along route:
{zone['days_from_origin']:.1f} days from origin

Retrieved evidence:

{chr(10).join(zone['documents'])}
"""
        )

    route_evidence = "\n".join(
        evidence_text
    )

    prompt = f"""
You are the geopolitical risk analysis engine
for RouteGuard, a maritime route risk prediction
system.

Analyze the geopolitical situation across the
ENTIRE maritime route.

Route:

Origin: {route['start']}

Destination: {route['end']}

Below is evidence retrieved from a vector database
for multiple monitoring zones along the route.

Determine ONE overall geopolitical risk score
for the complete route.

Return ONLY valid JSON:

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
- regional tensions that could realistically
  affect maritime transportation

Focus specifically on risks that could affect
the selected maritime route.

Do not treat ordinary political news as high risk
unless it has a meaningful connection to maritime
shipping or route disruption.

Retrieved route evidence:

{route_evidence}
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    text = response.text.strip()

    if text.startswith("```"):

        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()

    result = json.loads(text)

    risk_score = float(
        result["risk_score"]
    )

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


def analyze_route(
    start,
    end,
    n_zones=15,
    top_k=5
):

    print("\nROUTEGUARD GEOPOLITICAL ANALYSIS")

    print("\nGenerating maritime route...")

    route = generate_maritime_route(
        start,
        end
    )

    zones = create_risk_zones(
        route,
        n_zones=n_zones
    )

    print(
        f"Created {len(zones)} monitoring zones."
    )

    evidence = collect_route_evidence(
        zones,
        top_k=top_k
    )

    print("\nAnalyzing overall route situation...")

    route_info = {
        "start": start,
        "end": end,
        "distance_km": route.properties[
            "length"
        ],
        "duration_hours": route.properties[
            "duration_hours"
        ]
    }

    result = analyze_route_geopolitical_risk(
        route_info,
        evidence
    )

    print(
        f"\nOverall geopolitical risk: "
        f"{result['risk_score']}/10"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )

    return {
        "route": route_info,
        "zones": zones,
        "evidence": evidence,
        "geopolitical_risk_score": result[
            "risk_score"
        ],
        "geopolitical_reason": result[
            "reason"
        ]
    }


if __name__ == "__main__":

    start = [
        -0.1278,
        51.5074
    ]

    end = [
        121.4737,
        31.2304
    ]

    result = analyze_route(
        start,
        end,
        n_zones=15,
        top_k=5
    )

    print("\nROUTE SUMMARY")

    print(
        f"Distance: "
        f"{result['route']['distance_km']:.2f} km"
    )

    print(
        f"Duration: "
        f"{result['route']['duration_hours']:.2f} hours"
    )

    print(
        f"Geopolitical risk: "
        f"{result['geopolitical_risk_score']}/10"
    )