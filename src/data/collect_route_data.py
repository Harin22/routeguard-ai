import csv
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from data.route_sampler import generate_route_points
from data.weather import get_weather
from data.news import fetch_news, get_region


def collect_route_data(start, end, n_points=5):

    route_points = generate_route_points(
        start,
        end,
        n_points
    )

    rows = []

    for point_id, (lat, lon) in enumerate(route_points):

        print(f"\nCollecting Point {point_id}")
        print(f"Latitude: {lat}")
        print(f"Longitude: {lon}")

        # Weather
        weather = get_weather(lat, lon)

        # Geographic region
        region = get_region(lat, lon)

        # News
        news = fetch_news(region)

        row = {
            "point_id": point_id,
            "latitude": lat,
            "longitude": lon,
            "region": region,

            "temperature": weather.get("temperature"),
            "windspeed": weather.get("windspeed"),
            "winddirection": weather.get("winddirection"),
            "weathercode": weather.get("weathercode"),

            "news_count": len(news)
        }

        rows.append(row)

    return rows


def save_to_csv(rows, filename="route_points.csv"):

    fieldnames = [
        "point_id",
        "latitude",
        "longitude",
        "region",
        "temperature",
        "windspeed",
        "winddirection",
        "weathercode",
        "news_count"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDataset saved to: {filename}")


if __name__ == "__main__":

    # Shanghai → Rotterdam

    start = (31.23, 121.47)
    end = (51.92, 4.48)

    data = collect_route_data(
        start,
        end,
        n_points=5
    )

    save_to_csv(data)

    print("\nCollection complete.")