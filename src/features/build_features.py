import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.route_sampler import generate_route_points
from data.weather import get_weather
from data.news import get_news_signal


def weather_risk(w):
    risk = 0

    if w["windspeed"] > 25:
        risk += 0.6
    if w["weathercode"] > 50:
        risk += 0.4

    return min(risk, 1.0)


def evaluate_route(start, end, shift=0):
    start = (start[0] + shift, start[1])
    end = (end[0] + shift, end[1])

    points = generate_route_points(start, end, 5)

    total_weather = 0
    total_news = 0

    for lat, lon in points:
        w = get_weather(lat, lon)
        wr = weather_risk(w)

        total_weather += wr

        if wr < 0.5:
            news = get_news_signal(lat, lon)
            nr = news["conflict_risk"]
        else:
            nr = 0

        total_news += nr

    avg_weather = total_weather / len(points)
    avg_news = total_news / len(points)

    return round(avg_weather + avg_news, 3)


if __name__ == "__main__":
    start = (31.23, 121.47)
    end = (51.92, 4.48)

    routes = {
        "Route A": 0,
        "Route B": 2,
        "Route C": -2
    }

    results = {}

    for name, shift in routes.items():
        risk = evaluate_route(start, end, shift)
        results[name] = risk
        print(f"{name} → Risk: {risk}")

    best = min(results, key=results.get)

    print("\nBEST ROUTE:", best)