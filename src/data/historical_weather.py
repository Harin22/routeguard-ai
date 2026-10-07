import time
from pathlib import Path

import pandas as pd
import requests

from route_sampler import generate_maritime_route
from risk_zone import create_risk_zones


PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_URL = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

N_ZONES = 15

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "historical_weather_daily.csv"
)

HOURLY_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "rain",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "weather_code"
]


def fetch_historical_weather(latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "wind_speed_unit": "kn",
        "precipitation_unit": "mm",
        "timezone": "UTC",
        "cell_selection": "sea"
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


def build_route():

    start = [-0.1278, 51.5074]
    end = [121.4737, 31.2304]

    route = generate_maritime_route(
        start,
        end
    )

    zones = create_risk_zones(
        route,
        n_zones=N_ZONES
    )

    return zones


def collect_weather():

    print("\nROUTEGUARD DAILY WEATHER DATASET")
    print(
        f"Date range: "
        f"{START_DATE} → {END_DATE}"
    )

    zones = build_route()

    all_zone_data = []

    for zone in zones:

        zone_id = zone["zone_id"]

        print(
            f"\nFetching Zone {zone_id:02d}..."
        )

        print(
            f"Location: "
            f"{zone['latitude']:.4f}, "
            f"{zone['longitude']:.4f}"
        )

        weather = fetch_historical_weather(
            zone["latitude"],
            zone["longitude"]
        )

        hourly = pd.DataFrame(
            weather["hourly"]
        )

        hourly["time"] = pd.to_datetime(
            hourly["time"]
        )

        hourly["date"] = (
            hourly["time"].dt.date
        )

        daily = hourly.groupby("date").agg(
            temperature_2m=(
                "temperature_2m",
                "mean"
            ),
            precipitation=(
                "precipitation",
                "sum"
            ),
            rain=(
                "rain",
                "sum"
            ),
            pressure_msl=(
                "pressure_msl",
                "min"
            ),
            cloud_cover=(
                "cloud_cover",
                "mean"
            ),
            wind_speed_10m=(
                "wind_speed_10m",
                "max"
            ),
            wind_gusts_10m=(
                "wind_gusts_10m",
                "max"
            ),
            wind_direction_10m=(
                "wind_direction_10m",
                "mean"
            ),
            weather_code=(
                "weather_code",
                "max"
            )
        ).reset_index()

        daily["zone_id"] = zone_id

        all_zone_data.append(
            daily
        )

        print(
            f"Collected "
            f"{len(daily)} daily records."
        )

        time.sleep(1)

    zone_data = pd.concat(
        all_zone_data,
        ignore_index=True
    )

    print(
        "\nAggregating all 15 zones "
        "into daily route features..."
    )

    route_daily = zone_data.groupby("date").agg(
        temperature_2m=(
            "temperature_2m",
            "mean"
        ),
        precipitation=(
            "precipitation",
            "max"
        ),
        rain=(
            "rain",
            "max"
        ),
        pressure_msl=(
            "pressure_msl",
            "min"
        ),
        cloud_cover=(
            "cloud_cover",
            "mean"
        ),
        wind_speed_10m=(
            "wind_speed_10m",
            "max"
        ),
        wind_gusts_10m=(
            "wind_gusts_10m",
            "max"
        ),
        wind_direction_10m=(
            "wind_direction_10m",
            "mean"
        ),
        weather_code=(
            "weather_code",
            "max"
        )
    ).reset_index()

    route_daily["date"] = pd.to_datetime(
        route_daily["date"]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    route_daily.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print(
        f"\nRows: {len(route_daily)}"
    )

    print(
        f"Columns: {len(route_daily.columns)}"
    )

    print("\n=== PREVIEW ===")

    print(
        route_daily.head()
    )

    return route_daily


if __name__ == "__main__":
    collect_weather()