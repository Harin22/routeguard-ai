import os
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("NEWSDATA_API_KEY")

API_URL = "https://newsdata.io/api/1/latest"


def get_region(lat, lon):
    """
    Convert a route coordinate into a broad geographic region.
    """

    if 20 < lat < 40 and 30 < lon < 60:
        return "Middle East"

    elif 40 < lat < 60 and -10 < lon < 40:
        return "Europe"

    elif 5 < lat < 30 and 60 < lon < 100:
        return "South Asia"

    elif -10 < lat < 25 and 95 < lon < 130:
        return "Southeast Asia"

    elif 15 < lat < 45 and 100 < lon < 145:
        return "East Asia"

    elif 20 < lat < 50 and -130 < lon < -60:
        return "North America"

    else:
        return "Global"


def fetch_zone_news(lat, lon):
    """
    Fetch recent geopolitical and maritime-related
    news relevant to the zone.
    """

    if not API_KEY:
        raise ValueError(
            "NEWSDATA_API_KEY not found in .env"
        )

    region = get_region(lat, lon)

    query = (
        f"{region} war OR conflict OR military OR "
        f"attack OR tension OR security OR shipping"
    )

    params = {
        "apikey": API_KEY,
        "q": query,
        "language": "en",
        "size": 10
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    articles = data.get("results", [])

    return {
        "region": region,
        "articles": articles
    }


def attach_news_to_zones(zones):
    """
    Fetch news for every risk zone.
    """

    enriched_zones = []

    for zone in zones:

        print(
            f"Fetching news for "
            f"Zone {zone['zone_id']:02d}..."
        )

        news_data = fetch_zone_news(
            zone["latitude"],
            zone["longitude"]
        )

        zone_data = zone.copy()

        zone_data["region"] = news_data["region"]
        zone_data["news"] = news_data["articles"]

        enriched_zones.append(zone_data)

    return enriched_zones


if __name__ == "__main__":

    import searoute as sr
    from risk_zone import create_risk_zones

    # London → Shanghai
    start = [-0.1278, 51.5074]
    end = [121.4737, 31.2304]

    route = sr.searoute(
        start,
        end,
        units="km",
        append_orig_dest=True
    )

    zones = create_risk_zones(
        route,
        n_zones=15
    )

    enriched_zones = attach_news_to_zones(zones)

    print("\n=== NEWS TEST ===")

    for zone in enriched_zones:

        print(
            f"\nZone {zone['zone_id']:02d} | "
            f"Region: {zone['region']}"
        )

        print(
            f"Articles found: "
            f"{len(zone['news'])}"
        )

        for article in zone["news"][:3]:

            print(
                f"- {article.get('title', 'No title')}"
            )