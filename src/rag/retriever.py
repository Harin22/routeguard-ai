from pathlib import Path
import sys


# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# RouteGuard imports
# ---------------------------------------------------------

from src.data.risk_zone import create_risk_zones
from src.data.route_sampler import generate_maritime_route
from src.rag.vector_store import search_news


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

N_ZONES = 15
TOP_K = 5


# ---------------------------------------------------------
# Build test route
# ---------------------------------------------------------

def build_test_route():

    # London → Shanghai

    start = [
        -0.1278,
        51.5074
    ]

    end = [
        121.4737,
        31.2304
    ]

    route = generate_maritime_route(
        start,
        end
    )

    return route


# ---------------------------------------------------------
# Create automatic query for a zone
# ---------------------------------------------------------

def build_zone_query(zone):

    location_context = (
        f"maritime geopolitical risks "
        f"and shipping disruptions near "
        f"latitude {zone['latitude']:.4f}, "
        f"longitude {zone['longitude']:.4f}"
    )

    return location_context


# ---------------------------------------------------------
# Retrieve news for one zone
# ---------------------------------------------------------

def retrieve_zone_news(
    zone,
    top_k=TOP_K
):

    query = build_zone_query(
        zone
    )

    results = search_news(
        query=query,
        top_k=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    articles = []

    for i, document in enumerate(
        documents
    ):

        metadata = metadatas[i]

        articles.append({

            "title": metadata.get(
                "title",
                "Unknown"
            ),

            "source": metadata.get(
                "source",
                "Unknown"
            ),

            "published": metadata.get(
                "published",
                "Unknown"
            ),

            "location": metadata.get(
                "location",
                "Unknown"
            ),

            "zone_id": metadata.get(
                "zone_id",
                "Unknown"
            ),

            "distance": distances[i],

            "document": document
        })

    return {
        "zone": zone,
        "query": query,
        "articles": articles
    }


# ---------------------------------------------------------
# Retrieve news for entire route
# ---------------------------------------------------------

def retrieve_route_news():

    route = build_test_route()

    zones = create_risk_zones(
        route,
        n_zones=N_ZONES
    )

    route_results = []

    print(
        f"\nCreated {len(zones)} monitoring zones."
    )

    for zone in zones:

        print(
            f"\n================================"
        )

        print(
            f"ZONE {zone['zone_id']:02d}"
        )

        print(
            f"Location: "
            f"{zone['latitude']:.4f}, "
            f"{zone['longitude']:.4f}"
        )

        result = retrieve_zone_news(
            zone
        )

        print(
            f"Query: {result['query']}"
        )

        print(
            f"Retrieved: "
            f"{len(result['articles'])} articles"
        )

        route_results.append(
            result
        )

    return route_results


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "\n================================"
    )

    print(
        " ROUTEGUARD AUTOMATIC RETRIEVER"
    )

    print(
        "================================"
    )

    results = retrieve_route_news()

    print(
        "\n================================"
    )

    print(
        " RETRIEVAL COMPLETE"
    )

    print(
        "================================"
    )

    total_articles = sum(
        len(result["articles"])
        for result in results
    )

    print(
        f"\nZones processed: "
        f"{len(results)}"
    )

    print(
        f"Total retrieved results: "
        f"{total_articles}"
    )