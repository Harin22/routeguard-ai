from __future__ import annotations

from pathlib import Path
import sys
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.route_sampler import generate_maritime_route
from src.data.zone_news import fetch_zone_news
from src.data.risk_zone import create_risk_zones
from src.rag.vector_store import add_news_articles

N_ZONES = 15
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def build_route_key(start, end):
    """Create a unique analysis key so older news cannot leak into this run."""
    route = (
        f"{float(start[0]):.4f},{float(start[1]):.4f}>"
        f"{float(end[0]):.4f},{float(end[1]):.4f}"
    )
    return f"{route}|analysis:{uuid.uuid4().hex}"


def fetch_zone_news_with_retry(latitude, longitude):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_zone_news(latitude, longitude)
        except Exception as error:
            last_error = error
            print(f"News request failed (attempt {attempt}/{MAX_RETRIES}): {error}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    print(f"Zone news unavailable after retries: {last_error}")
    return None


def collect_route_news(start, end, route_key=None, n_zones=N_ZONES):
    """Fetch fresh provider news from monitoring points on this selected route."""
    route_key = route_key or build_route_key(start, end)
    route = generate_maritime_route(start, end)
    zones = create_risk_zones(route, n_zones=n_zones)
    unique_articles = {}
    zones_with_articles = 0
    failed_zones = []
    raw_articles_seen = 0

    print(f"Fetching fresh news for {len(zones)} zones on route {route_key}")
    for zone in zones:
        zone_id = zone["zone_id"]
        news_data = fetch_zone_news_with_retry(zone["latitude"], zone["longitude"])
        if news_data is None:
            failed_zones.append(zone_id)
            continue

        articles = news_data.get("articles", []) or []
        if articles:
            zones_with_articles += 1
        for article in articles:
            if article.get("duplicate") is True:
                continue
            raw_articles_seen += 1
            article_id = article.get("article_id")
            article_url = article.get("link")
            stable_id = str(article_id).strip() if article_id else str(article_url or "").strip().rstrip("/")
            if not stable_id:
                continue

            if stable_id not in unique_articles:
                unique_articles[stable_id] = {
                    "article_id": article_id or "",
                    "link": article_url or "",
                    "title": article.get("title", "") or "",
                    "description": article.get("description", "") or "",
                    "content": article.get("content", "") or "",
                    "source": article.get("source_name", "unknown") or "unknown",
                    "published": article.get("pubDate", "") or "",
                    "location": news_data.get("location", "Unknown"),
                    "zone_id": zone_id,
                    "zone_ids": [str(zone_id)],
                    "route_key": route_key,
                }
            else:
                existing = unique_articles[stable_id]
                if str(zone_id) not in existing["zone_ids"]:
                    existing["zone_ids"].append(str(zone_id))
                for field, source_field in (
                    ("title", "title"),
                    ("description", "description"),
                    ("content", "content"),
                    ("source", "source_name"),
                    ("published", "pubDate"),
                    ("link", "link"),
                ):
                    if not existing.get(field) and article.get(source_field):
                        existing[field] = article[source_field]

    return {
        "articles": list(unique_articles.values()),
        "zones_checked": len(zones),
        "zones_with_articles": zones_with_articles,
        "failed_zones": failed_zones,
        "raw_articles_seen": raw_articles_seen,
        "route_key": route_key,
    }


def ingest_route_news(start, end, route_key=None, n_zones=N_ZONES):
    """Fetch and embed route-specific fresh news before route risk analysis."""
    batch = collect_route_news(start, end, route_key=route_key, n_zones=n_zones)
    articles = batch["articles"]
    if not articles:
        failed = batch["failed_zones"]
        raise RuntimeError(
            "No fresh news articles were collected for this route. "
            f"Failed zones: {failed or 'none reported'}. "
            "Route analysis stopped to avoid using unrelated/stale news."
        )

    add_news_articles(articles)
    return {
        "route_key": batch["route_key"],
        "zones_checked": batch["zones_checked"],
        "zones_with_articles": batch["zones_with_articles"],
        "zones_failed": batch["failed_zones"],
        "raw_articles_seen": batch["raw_articles_seen"],
        "unique_articles_upserted": len(articles),
    }


def main():
    start = [-0.1278, 51.5074]
    end = [121.4737, 31.2304]
    summary = ingest_route_news(start, end)
    print("News ingestion complete:", summary)


if __name__ == "__main__":
    main()
