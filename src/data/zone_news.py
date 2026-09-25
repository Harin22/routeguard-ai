import os
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("NEWSDATA_API_KEY")

API_URL = "https://newsdata.io/api/1/latest"


def get_location_name(lat, lon):
    """
    Convert route coordinates into a useful
    maritime/geographic location label.
    """

    # Europe / English Channel
    if lat > 48 and -10 < lon < 5:
        return "English Channel Europe"

    # Mediterranean
    elif 30 < lat < 45 and -6 < lon < 35:
        return "Mediterranean Sea"

    # Suez / Red Sea
    elif 10 < lat < 32 and 30 < lon < 45:
        return "Red Sea Suez"

    # Arabian Sea / Indian Ocean
    elif 5 < lat < 25 and 45 < lon < 80:
        return "Arabian Sea Indian Ocean"

    # Bay of Bengal
    elif 5 < lat < 22 and 80 < lon < 100:
        return "Bay of Bengal"

    # Southeast Asia / Malacca
    elif -5 < lat < 15 and 95 < lon < 110:
        return "Strait of Malacca Southeast Asia"

    # South China Sea
    elif 5 < lat < 25 and 105 < lon < 120:
        return "South China Sea"

    # East China Sea
    elif 20 < lat < 40 and 120 < lon < 135:
        return "East China Sea"

    # North Pacific
    elif lat > 25 and 135 < lon < 180:
        return "North Pacific"

    # North Atlantic
    elif lat > 30 and -60 < lon < -10:
        return "North Atlantic"

    else:
        return "Global Maritime"


def fetch_zone_news(lat, lon):
    """
    Fetch recent geopolitical and maritime news
    relevant to the route zone.
    """

    if not API_KEY:
        raise ValueError(
            "NEWSDATA_API_KEY not found in .env"
        )

    location = get_location_name(
        lat,
        lon
    )

    query = location

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

    articles = data.get(
        "results",
        []
    )

    return {
        "location": location,
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

        zone_data["news_location"] = (
            news_data["location"]
        )

        zone_data["news"] = (
            news_data["articles"]
        )

        enriched_zones.append(
            zone_data
        )

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

    enriched_zones = attach_news_to_zones(
        zones
    )

    print("\n=== NEWS TEST ===")

    for zone in enriched_zones:

        print(
            f"\nZone {zone['zone_id']:02d} | "
            f"Location: "
            f"{zone['news_location']}"
        )

        print(
            f"Articles found: "
            f"{len(zone['news'])}"
        )

        for article in zone["news"][:3]:

            print(
                f"- "
                f"{article.get('title', 'No title')}"
            )