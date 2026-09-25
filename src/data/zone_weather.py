import requests


API_URL = "https://api.open-meteo.com/v1/forecast"


def get_zone_weather(latitude, longitude):
    """
    Get 7-day hourly weather forecast for one risk zone.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": 7,
        "hourly": (
            "temperature_2m,"
            "precipitation,"
            "precipitation_probability,"
            "windspeed_10m,"
            "winddirection_10m,"
            "windgusts_10m,"
            "weathercode"
        ),
        "wind_speed_unit": "kn",
        "timezone": "UTC",
        "cell_selection": "sea"
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    return response.json()


def attach_weather_to_zones(zones):
    """
    Fetch weather for every risk zone.

    Each zone receives its 7-day hourly forecast.
    """

    enriched_zones = []

    for zone in zones:

        print(
            f"Fetching weather for "
            f"Zone {zone['zone_id']:02d}..."
        )

        weather = get_zone_weather(
            zone["latitude"],
            zone["longitude"]
        )

        zone_data = zone.copy()

        zone_data["weather"] = weather["hourly"]

        enriched_zones.append(zone_data)

    return enriched_zones


if __name__ == "__main__":

    import searoute as sr
    from risk_zone import create_risk_zones

    # Shanghai → Rotterdam
    start = [121.4737, 31.2304]
    end = [4.4777, 51.9244]

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

    enriched_zones = attach_weather_to_zones(zones)

    print("\n=== Weather results.. ===")

    for zone in enriched_zones:

        weather = zone["weather"]

        print(
            f"\nZone {zone['zone_id']:02d} | "
            f"Lat={zone['latitude']:.4f} | "
            f"Lon={zone['longitude']:.4f}"
        )

        print(
            f"Forecast hours: "
            f"{len(weather['time'])}"
        )

        print(
            f"Temperature: "
            f"{weather['temperature_2m'][0]} °C"
        )

        print(
            f"Wind: "
            f"{weather['windspeed_10m'][0]} kn"
        )

        print(
            f"Wind gust: "
            f"{weather['windgusts_10m'][0]} kn"
        )

        print(
            f"Precipitation: "
            f"{weather['precipitation'][0]} mm"
        )