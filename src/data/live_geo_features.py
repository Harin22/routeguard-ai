from datetime import datetime, timedelta, timezone
import time

import requests

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def get_live_geo_features(latitude, longitude, radius_km=250):
    """Fetch recent GDELT news-based geopolitical indicators."""

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    params = {
        "query": (
            f"near:{latitude},{longitude},{radius_km} "
            "(war OR attack OR conflict OR piracy OR security)"
        ),
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "maxrecords": 50,
        "sort": "DateDesc",
    }

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "RouteGuardAI/1.0 (research project)"
        })

        response = None

        for attempt in range(3):
            try:
                response = session.get(
                    GDELT_API,
                    params=params,
                    timeout=30,
                )

                if response.status_code == 429:
                    if attempt < 2:
                        wait = 30 * (attempt + 1)
                        print(f"GDELT rate limited. Waiting {wait}s...")
                        time.sleep(wait)
                        continue

                    raise RuntimeError(
                        "GDELT rate limit persists. Wait several minutes "
                        "before trying again."
                    )

                response.raise_for_status()
                break

            except requests.RequestException as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"GDELT request failed: {exc}"
                    ) from exc

                time.sleep(5 * (attempt + 1))

    data = response.json()
    articles = data.get("articles", [])

    hostile_terms = (
        "attack", "strike", "war", "piracy",
        "hostile", "missile", "armed conflict",
    )

    hostile_events = sum(
        any(term in article.get("title", "").lower()
            for term in hostile_terms)
        for article in articles
    )

    tones = []
    for article in articles:
        tone = article.get("tone")
        if tone is not None:
            try:
                tones.append(float(tone))
            except (TypeError, ValueError):
                pass

    return {
        "geo_security_events": len(articles),
        "geo_hostile_events": hostile_events,
        "geo_avg_tone": (
            sum(tones) / len(tones) if tones else 0.0
        ),
        "articles_found": len(articles),
        "source": "GDELT news-document proxy",
    }


if __name__ == "__main__":
    try:
        result = get_live_geo_features(12.0, 45.0)
        print(result)
    except (RuntimeError, ValueError) as exc:
        print(f"Geopolitical data unavailable: {exc}")