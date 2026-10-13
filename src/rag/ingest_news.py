
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

    start = [-0.1278, 51.5074]
    end = [121.4737, 31.2304]

    return sr.searoute(
        start,
        end,
        units="km",
        append_orig_dest=True,
    )


# ---------------------------------------------------------
# Fetch news with retry
# ---------------------------------------------------------

def fetch_zone_news_with_retry(latitude, longitude):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_zone_news(latitude, longitude)

        except Exception as error:
            print(
                f"Request failed (attempt "
                f"{attempt}/{MAX_RETRIES}): {error}"
            )

            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print("Maximum retries reached. Skipping this zone.")

    return None


# ---------------------------------------------------------
# Collect and deduplicate route news
# ---------------------------------------------------------

def collect_route_news():
    route = build_test_route()

    zones = create_risk_zones(
        route,
        n_zones=N_ZONES,
    )

    # Keyed by stable article identity to deduplicate across zones.
    unique_articles = {}

    print(f"\nCreated {len(zones)} monitoring zones.")

    for zone in zones:
        zone_id = zone["zone_id"]

        print(f"\nFetching news for Zone {zone_id:02d}...")

        news_data = fetch_zone_news_with_retry(
            zone["latitude"],
            zone["longitude"],
        )

        if news_data is None:
            print(f"Zone {zone_id:02d} skipped.")
            continue

        location = news_data["location"]
        articles = news_data["articles"]

        print(f"Location: {location}")
        print(f"Articles found: {len(articles)}")

        for article in articles:
            # Skip only when the API explicitly marks the value
            # as a duplicate. Do not treat a missing value as True.
            if article.get("duplicate") is True:
                continue

            article_id = article.get("article_id")
            article_url = article.get("link")

            if article_id:
                stable_id = str(article_id).strip()
            elif article_url:
                stable_id = str(article_url).strip().rstrip("/")
            else:
                print("Skipping article without an ID or URL.")
                continue

            if not stable_id:
                continue

            if stable_id not in unique_articles:
                unique_articles[stable_id] = {
                    "id": stable_id,
                    "article_id": article_id or "",
                    "link": article_url or "",
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", ""),
                    "source": article.get("source_name", "unknown"),
                    "published": article.get("pubDate", ""),
                    "location": location,
                    "zone_id": zone_id,
                    "zone_ids": [str(zone_id)],
                }
            else:
                # Preserve every zone where this article appeared.
                existing = unique_articles[stable_id]

                if str(zone_id) not in existing["zone_ids"]:
                    existing["zone_ids"].append(str(zone_id))

                # Fill missing content or metadata from another result.
                for field in (
                    "title",
                    "description",
                    "content",
                    "source",
                    "published",
                    "link",
                ):
                    if not existing.get(field) and article.get(field):
                        source_field = {
                            "source": "source_name",
                            "published": "pubDate",
                            "link": "link",
                        }.get(field, field)

                        existing[field] = article.get(source_field, "")

    all_articles = list(unique_articles.values())

    print(f"\nUnique articles collected: {len(all_articles)}")
    print(f"Duplicates skipped across zones: {sum(1 for _ in [])}")

    return all_articles


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("\n================================")
    print(" ROUTEGUARD NEWS INGESTION")
    print("================================")

    articles = collect_route_news()

    print(f"\nTotal unique articles collected: {len(articles)}")

    if not articles:
        print("\nNo articles were collected.")
        return

    print("\nSending articles to Cohere + ChromaDB...")

    add_news_articles(articles)

    print("\n================================")
    print(" INGESTION COMPLETE")
    print("================================")


if __name__ == "__main__":
    main()
