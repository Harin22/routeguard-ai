import json
import os
import sys
import requests
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
MERCURY_API_KEY = os.getenv("MERCURY_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

if not MERCURY_API_KEY:
    raise ValueError("MERCURY_API_KEY not found in .env")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.8-flash"
MERCURY_MODEL = "mercury-2.5"
MERCURY_URL = "https://api.inceptionlabs.ai/v1/chat/completions"

from src.data.route_sampler import generate_maritime_route
from src.data.risk_zone import create_risk_zones

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "routeguard_news"


class CohereEmbeddings(Embeddings):

    def __init__(self):
        import cohere

        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in .env")

        self.client = cohere.ClientV2(api_key=cohere_api_key)

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


def retrieve_news_for_zone(latitude, longitude, top_k=5):
    query = (
        "maritime geopolitical risks and "
        "shipping disruptions near "
        f"latitude {latitude:.4f}, "
        f"longitude {longitude:.4f}"
    )

    return vector_store.similarity_search(query, k=top_k)


def collect_route_evidence(zones, top_k=5):
    evidence = []

    for zone in zones:
        print(f"\nZone {zone['zone_id']:02d}")
        print(
            f"Location: {zone['latitude']:.4f}, "
            f"{zone['longitude']:.4f}"
        )
        print("Retrieving news...")

        documents = retrieve_news_for_zone(
            zone["latitude"],
            zone["longitude"],
            top_k=top_k
        )

        print(f"Retrieved {len(documents)} documents.")

        evidence.append({
            "zone_id": zone["zone_id"],
            "latitude": zone["latitude"],
            "longitude": zone["longitude"],
            "days_from_origin": zone["days_from_origin"],
            "documents": [
                document.page_content
                for document in documents
            ]
        })

    return evidence


def call_gemini(prompt):
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    return response.text.strip()


def call_mercury(prompt):
    response = requests.post(
        MERCURY_URL,
        headers={
            "Authorization": f"Bearer {MERCURY_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MERCURY_MODEL,
            "reasoning_effort": "low",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        timeout=60
    )

    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]

    if not content:
        raise ValueError("Mercury returned an empty response")

    return content.strip()


def parse_risk_response(response_text):
    text = response_text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "", 1)
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    result = json.loads(text)
    risk_score = float(result["risk_score"])

    if not 0 <= risk_score <= 10:
        raise ValueError("Risk score must be between 0 and 10")

    reason = result.get("reason", "").strip()

    if not reason:
        raise ValueError("Risk explanation is missing")

    return {
        "risk_score": risk_score,
        "reason": reason
    }


def analyze_route_geopolitical_risk(route, evidence):
    evidence_text = []

    for zone in evidence:
        evidence_text.append(
            f"""
ZONE {zone['zone_id']}
Location: {zone['latitude']:.4f}, {zone['longitude']:.4f}
Approximate position: {zone['days_from_origin']:.1f} days from origin

Retrieved evidence:
{chr(10).join(zone['documents'])}
"""
        )

    route_evidence = "\n".join(evidence_text)

    prompt = f"""
You are RouteGuard's maritime geopolitical risk analysis engine.

Analyze the evidence for the entire maritime route.

Origin: {route['start']}
Destination: {route['end']}
Distance: {route['distance_km']} km
Estimated duration: {route['duration_hours']} hours

Assess armed conflict, attacks on ships, blockades, piracy,
sanctions, political instability, and disruptions that could
realistically affect maritime shipping.

Do not treat ordinary political news as high risk without a
meaningful connection to maritime shipping.

Use the retrieved evidence. Do not invent events or facts.
If evidence is weak or missing, reflect that uncertainty.

Return ONLY valid JSON in this format:
{{
  "risk_score": 0,
  "reason": "Short explanation based on the evidence"
}}

Score: 0 means minimal identified risk; 10 means extremely
severe identified risk. This is a risk index, not a calibrated
probability of disruption.

Retrieved route evidence:
{route_evidence}
"""

    try:
        print("Trying Gemini...")
        response_text = call_gemini(prompt)
        result = parse_risk_response(response_text)
        result["provider"] = "Gemini"
        print("Gemini analysis succeeded.")

    except Exception as gemini_error:
        print(f"Gemini failed: {gemini_error}")
        print("Switching to Mercury...")

        try:
            response_text = call_mercury(prompt)
            result = parse_risk_response(response_text)
            result["provider"] = "Mercury"
            print("Mercury analysis succeeded.")

        except Exception as mercury_error:
            raise RuntimeError(
                "Both Gemini and Mercury failed. "
                f"Gemini: {gemini_error}; Mercury: {mercury_error}"
            ) from mercury_error

    return result


def analyze_route(start, end, n_zones=15, top_k=5):
    print("\nROUTEGUARD GEOPOLITICAL ANALYSIS")
    print("\nGenerating maritime route...")

    route = generate_maritime_route(start, end)
    zones = create_risk_zones(route, n_zones=n_zones)

    print(f"Created {len(zones)} monitoring zones.")

    evidence = collect_route_evidence(zones, top_k=top_k)

    print("\nAnalyzing overall route situation...")

    route_info = {
        "start": start,
        "end": end,
        "distance_km": route.properties["length"],
        "duration_hours": route.properties["duration_hours"]
    }

    result = analyze_route_geopolitical_risk(route_info, evidence)

    print(f"\nOverall geopolitical risk: {result['risk_score']}/10")
    print(f"Reason: {result['reason']}")
    print(f"Provider: {result['provider']}")

    return {
        "route": route_info,
        "zones": zones,
        "evidence": evidence,
        "geopolitical_risk_score": result["risk_score"],
        "geopolitical_reason": result["reason"],
        "analysis_provider": result["provider"]
    }


if __name__ == "__main__":
    start = [-0.1278, 51.5074]
    end = [121.4737, 31.2304]

    result = analyze_route(
        start,
        end,
        n_zones=15,
        top_k=5
    )

    print("\nROUTE SUMMARY")
    print(f"Distance: {result['route']['distance_km']:.2f} km")
    print(f"Duration: {result['route']['duration_hours']:.2f} hours")
    print(f"Geopolitical risk: {result['geopolitical_risk_score']}/10")
    print(f"Analysis provider: {result['analysis_provider']}")
