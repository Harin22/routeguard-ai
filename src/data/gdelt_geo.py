
from collections import Counter
from datetime import timedelta
from pathlib import Path
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/raw/gdelt_sample.zip"
OUTPUT = ROOT / "data/processed/geopolitical_daily.csv"

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")
DOWNLOAD_END = END + pd.Timedelta(days=7)

# Main maritime risk areas along London–Shanghai
ROUTE_REGIONS = [
    (28, 32, 31, 35),       # Suez Canal
    (11, 15, 41, 46),       # Bab el-Mandeb
    (-2, 8, 98, 105),       # Malacca Strait
    (5, 25, 105, 125),      # South China Sea
]

# CAMEO event roots: security-related activity and hostile events
SECURITY_ROOTS = set(range(13, 21))
HOSTILE_ROOTS = {17, 18, 19, 20}

def get_archive(day, session):
    # Reuse the sample already downloaded
    if day == pd.Timestamp("2025-01-01") and SAMPLE.exists():
        return SAMPLE.read_bytes()

    url = (
        "https://data.gdeltproject.org/events/"
        f"{day:%Y%m%d}.export.CSV.zip"
    )

    for attempt in range(3):
        try:
            response = session.get(url, timeout=90)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            if attempt == 2:
                print(f"Download failed for {day.date()}: {exc}")
                return None
            time.sleep(attempt + 1)

def process_archive(day, payload, outcomes):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_name = next(
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        )

        with archive.open(csv_name) as file:
            df = pd.read_csv(
                file,
                sep="\t",
                header=None,
                usecols=[1, 28, 34, 53, 54],
                dtype={1: "string", 28: "string"},
                low_memory=False,
            )

    root = pd.to_numeric(df[28], errors="coerce")
    tone = pd.to_numeric(df[34], errors="coerce")
    lat = pd.to_numeric(df[53], errors="coerce")
    lon = pd.to_numeric(df[54], errors="coerce")

    # Events geographically near the monitored sea corridors
    near = np.zeros(len(df), dtype=bool)
    for lat1, lat2, lon1, lon2 in ROUTE_REGIONS:
        near |= (
            lat.between(lat1, lat2)
            & lon.between(lon1, lon2)
        ).to_numpy()

    security = near & root.isin(SECURITY_ROOTS).to_numpy()
    hostile = near & root.isin(HOSTILE_ROOTS).to_numpy()

    # Retrospective event dates are used for proxy outcome labels.
    event_date = pd.to_datetime(
        df[1].str[:8], format="%Y%m%d", errors="coerce"
    )
    target_dates = event_date[hostile].dropna()
    target_dates = target_dates[
        target_dates.between(START, DOWNLOAD_END)
    ]

    for event_day, count in target_dates.dt.normalize().value_counts().items():
        outcomes[pd.Timestamp(event_day)] += int(count)

    return {
        "date": day,
        "geo_security_events": int(security.sum()),
        "geo_hostile_events": int(hostile.sum()),
        "geo_avg_tone": (
            float(tone[security].mean())
            if security.any() else 0.0
        ),
        "archive_available": 1,
    }

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(START, DOWNLOAD_END, freq="D")
    records = []
    outcomes = Counter()
    session = requests.Session()
    session.headers.update({"User-Agent": "RouteGuardAI/1.0"})

    for i, day in enumerate(dates, start=1):
        payload = get_archive(day, session)

        if payload is None:
            records.append({
                "date": day,
                "geo_security_events": np.nan,
                "geo_hostile_events": np.nan,
                "geo_avg_tone": np.nan,
                "archive_available": 0,
            })
        else:
            try:
                records.append(
                    process_archive(day, payload, outcomes)
                )
            except Exception as exc:
                print(f"Processing failed for {day.date()}: {exc}")
                records.append({
                    "date": day,
                    "geo_security_events": np.nan,
                    "geo_hostile_events": np.nan,
                    "geo_avg_tone": np.nan,
                    "archive_available": 0,
                })

        if i % 10 == 0 or i == len(dates):
            print(f"Processed {i}/{len(dates)} days")

    daily = (
        pd.DataFrame(records)
        .set_index("date")
        .reindex(dates)
    )
    daily.index.name = "date"

    # Historical event occurrence proxy for target generation
    daily["geo_hazard_event_count"] = [
        outcomes.get(day, 0) for day in dates
    ]
    daily["geo_hazard_day"] = (
        daily["geo_hazard_event_count"] > 0
    ).astype(float)
    daily.loc[daily["archive_available"] == 0, "geo_hazard_day"] = np.nan

    # Inputs available from the current and previous six days
    daily["geo_security_events_7d"] = (
        daily["geo_security_events"].rolling(7, min_periods=7).sum()
    )
    daily["geo_hostile_events_7d"] = (
        daily["geo_hostile_events"].rolling(7, min_periods=7).sum()
    )
    daily["geo_avg_tone_7d"] = (
        daily["geo_avg_tone"].rolling(7, min_periods=7).mean()
    )

    # Transparent heuristic index, NOT Gemini's score
    daily["geo_risk_score"] = (
        np.log1p(daily["geo_hostile_events_7d"])
        + (-daily["geo_avg_tone_7d"]).clip(0, 10) / 10
    ).clip(0, 10)

    daily.reset_index().to_csv(OUTPUT, index=False)

    missing = int((daily["archive_available"] == 0).sum())
    print(f"Rows: {len(daily)}")
    print(f"Unavailable archives: {missing}")
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()
