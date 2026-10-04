from pathlib import Path
import sys
import time

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# RouteGuard imports
# ---------------------------------------------------------

from src.data.zone_news import fetch_zone_news
from src.data.risk_zone import create_risk_zones
from src.rag.vector_store import add_news_articles


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

N_ZONES = 15
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3


# ---------------------------------------------------------
# Generate test route
# ---------------------------------------------------------

def build_test_route():

    import searoute as sr

    # London → Shanghai
    start = [
        -0.1278,
        51.5074
    ]

    end = [
        121.4737,
        31.2304
    ]

    route = sr.searoute(
        start,
        end,
        units="km",
        append_orig_dest=True
    )

    return route


# ---------------------------------------------------------
# Fetch news with retry
# ---------------------------------------------------------

def fetch_zone_news_with_retry(
    latitude,
    longitude
):

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            return fetch_zone_news(
                latitude,
                longitude
            )

        except Exception as error:

            print(
                f"Request failed "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            print(
                f"Reason: {error}"
            )

            if attempt < MAX_RETRIES:

                print(
                    f"Retrying in "
                    f"{RETRY_DELAY_SECONDS} seconds..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

            else:

                print(
                    "Maximum retries reached. "
                    "Skipping this zone."
                )

    return None


# ---------------------------------------------------------
# Collect news from route zones
# ---------------------------------------------------------

def collect_route_news():

    route = build_test_route()

    zones = create_risk_zones(
        route,
        n_zones=N_ZONES
    )

    all_articles = []

    print(
        f"\nCreated {len(zones)} monitoring zones."
    )

    for zone in zones:

        zone_id = zone["zone_id"]
        latitude = zone["latitude"]
        longitude = zone["longitude"]

        print(
            f"\nFetching news for Zone "
            f"{zone_id:02d}..."
        )

        news_data = fetch_zone_news_with_retry(
            latitude,
            longitude
        )

        # If this zone failed after retries,
        # continue with the remaining zones.
        if news_data is None:

            print(
                f"Zone {zone_id:02d} skipped."
            )

            continue

        location = news_data["location"]
        articles = news_data["articles"]

        print(
            f"Location: {location}"
        )

        print(
            f"Articles found: {len(articles)}"
        )

        for article in articles:

            article_data = {
                "id": (
                    f"zone-{zone_id}-"
                    f"{article.get('article_id', '')}"
                ),

                "title": article.get(
                    "title",
                    ""
                ),

                "description": article.get(
                    "description",
                    ""
                ),

                "content": article.get(
                    "content",
                    ""
                ),

                "source": article.get(
                    "source_name",
                    "unknown"
                ),

                "published": article.get(
                    "pubDate",
                    ""
                ),

                "location": location,

                "zone_id": zone_id,
            }

            all_articles.append(
                article_data
            )

    return all_articles


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print(
        "\n================================"
    )

    print(
        " ROUTEGUARD NEWS INGESTION"
    )

    print(
        "================================"
    )

    # Collect news
    articles = collect_route_news()

    print(
        f"\nTotal articles collected: "
        f"{len(articles)}"
    )

    if not articles:

        print(
            "\nNo articles were collected."
        )

        return

    # Create embeddings + store in ChromaDB
    print(
        "\nSending articles to "
        "Cohere + ChromaDB..."
    )

    add_news_articles(
        articles
    )

    print(
        "\n================================"
    )

    print(
        " INGESTION COMPLETE"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()