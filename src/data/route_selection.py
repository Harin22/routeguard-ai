import joblib
import numpy as np
import pandas as pd
import searoute as sr
from pathlib import Path

from src.data.route_sampler import (
    build_route,
    generate_alternative_routes,
)
from src.data.risk_zone import create_risk_zones
from src.data.zone_weather import attach_weather_to_zones
from src.rag.geopolitical_analyzer import analyze_route

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "logistic_regression.joblib"

WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "rain",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "weather_code",
]

GEO_FEATURES = [
    "geo_security_events",
    "geo_hostile_events",
    "geo_avg_tone",
]

FEATURES = WEATHER_FEATURES + GEO_FEATURES


def predict_route_risk(route, geopolitical_result):
    """Predict weather hazard risk, fuse with the geo index, and expose zone details."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    zones = create_risk_zones(route, n_zones=15)
    zones = attach_weather_to_zones(zones)

    zone_rows = []
    weather_zone_details = []

    for zone in zones:
        hourly = zone["weather"]
        row = {}

        for feature in WEATHER_FEATURES:
            values = np.asarray(hourly[feature][:24], dtype=float)
            if values.size == 0 or not np.isfinite(values).any():
                raise ValueError(
                    f"Missing weather data for {feature}, zone {zone['zone_id']}"
                )
            values = values[np.isfinite(values)]
            if feature in ("precipitation", "rain"):
                row[feature] = float(values.sum())
            else:
                row[feature] = float(values.mean())

        zone_rows.append(row)
        weather_zone_details.append({
            "zone_id": zone["zone_id"],
            "latitude": zone["latitude"],
            "longitude": zone["longitude"],
            "forecast_time_utc": (hourly.get("time") or ["Unknown"])[0],
            **row,
            "weather_code_now": (hourly.get("weather_code") or [None])[0],
        })

    weather_summary = pd.DataFrame(zone_rows).mean()
    model_input = {
        feature: float(weather_summary[feature])
        for feature in WEATHER_FEATURES
    }
    # The trained pipeline imputes absent historical geo features using training medians.
    # The live geopolitical score is fused separately below.
    model_input.update({feature: np.nan for feature in GEO_FEATURES})

    X = pd.DataFrame([model_input], columns=FEATURES)
    probabilities = model.predict_proba(X)[0]
    class_index = list(model.classes_).index(1)
    ml_probability = float(probabilities[class_index])
    ml_score = ml_probability * 10
    geopolitical_score = float(geopolitical_result["geopolitical_risk_score"])
    final_score = (ml_score + geopolitical_score) / 2

    return {
        "ml_hazard_probability": ml_probability,
        "ml_risk_score": ml_score,
        "geopolitical_risk_score": geopolitical_score,
        "final_risk_score": final_score,
        "weather_zones_checked": len(zones),
        "weather_zone_details": weather_zone_details,
        "geopolitical_reason": geopolitical_result["geopolitical_reason"],
        "analysis_provider": geopolitical_result["analysis_provider"],
        "geopolitical_evidence": geopolitical_result.get("evidence", []),
    }


def select_route(start, end, risk_threshold=6.0):
    """Run fresh route-news ingestion, geopolitics, weather, and model fusion."""
    primary = build_route(start, end)

    # Fresh news must be ingested before geopolitical analysis retrieves evidence.
    from src.rag.ingest_news import build_route_key, ingest_route_news
    route_key = build_route_key(start, end)
    news_ingestion = ingest_route_news(start, end, route_key=route_key, n_zones=15)
    geopolitical_result = analyze_route(start, end, n_zones=15, top_k=5, route_key=route_key)
    route = sr.searoute(
        start,
        end,
        units="km",
        append_orig_dest=True,
    )

    risk = predict_route_risk(route, geopolitical_result)
    risk_score = risk["final_risk_score"]
    result = {
        "primary_route": primary,
        "route_key": route_key,
        "news_ingestion": news_ingestion,
        **risk,
        "risk_threshold": risk_threshold,
    }

    if risk_score < risk_threshold:
        result["status"] = "primary_route_recommended"
        result["message"] = "Primary route is below the risk threshold."
        result["alternatives"] = []
        return result

    candidates = generate_alternative_routes(start, end)
    primary_geometry = tuple(
        (round(lon, 4), round(lat, 4)) for lon, lat in primary["full_route"]
    )
    alternatives = []
    seen = {primary_geometry}

    for candidate in candidates:
        geometry = tuple(
            (round(lon, 4), round(lat, 4))
            for lon, lat in candidate["full_route"]
        )
        if geometry in seen:
            continue
        seen.add(geometry)
        alternatives.append(candidate)

    alternatives.sort(key=lambda item: item["distance_km"])
    result["status"] = "alternatives_offered"
    result["message"] = "Risk reached the threshold. Choose an available alternative."
    result["alternatives"] = [
        {"option": index, **candidate}
        for index, candidate in enumerate(alternatives[:2], start=1)
    ]
    if not result["alternatives"]:
        result["message"] += " No distinct alternatives were found."
    return result


if __name__ == "__main__":
    start = [-0.1278, 51.5074]
    end = [121.4737, 31.2304]
    result = select_route(start, end)
    print("\n=== ROUTEGUARD: LONDON TO SHANGHAI ===")
    print("Status:", result["status"])
    print(f"ML hazard probability: {result['ml_hazard_probability']:.1%}")
    print(f"ML risk score: {result['ml_risk_score']:.2f}/10")
    print(f"Geopolitical risk: {result['geopolitical_risk_score']:.2f}/10")
    print(f"Fused risk: {result['final_risk_score']:.2f}/10")
    print(f"Weather zones: {result['weather_zones_checked']}")
    print("Provider:", result["analysis_provider"])
    print("Reason:", result["geopolitical_reason"])
    print(f"Primary distance: {result['primary_route']['distance_km']:.2f} km")
    for alternative in result["alternatives"]:
        print(f"Alternative {alternative['option']}: {alternative['distance_km']:.2f} km")
