
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

WEATHER_FILE = ROOT / "data/historical_weather_daily.csv"
GEO_FILE = ROOT / "data/processed/geopolitical_daily.csv"
OUTPUT_FILE = ROOT / "data/processed/training_dataset.csv"

weather = pd.read_csv(WEATHER_FILE, parse_dates=["date"])
geo = pd.read_csv(GEO_FILE, parse_dates=["date"])

# Keep the complete weather calendar; preserve missing geo dates as missing.
df = weather.merge(geo, on="date", how="left")
df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

required_geo = [
    "archive_available",
    "geo_hazard_event_count",
    "geo_security_events",
    "geo_hostile_events",
    "geo_avg_tone",
    "geo_security_events_7d",
    "geo_hostile_events_7d",
    "geo_avg_tone_7d",
    "geo_risk_score",
]

# A day is usable only when its geo archive and features are available.
df["geo_available"] = (
    df["archive_available"].eq(1)
    & df[required_geo[1:]].notna().all(axis=1)
)

storm_codes = {65, 75, 82, 86, 95, 96, 99}
df["severe_weather_day"] = (
    df["wind_gusts_10m"].ge(60)
    | df["weather_code"].isin(storm_codes)
).astype(int)

# Fit the high-activity threshold on the earlier 80% only.
split_date = df["date"].iloc[int(len(df) * 0.8)]
threshold = df.loc[
    (df["date"] < split_date) & df["geo_available"],
    "geo_hazard_event_count",
].quantile(0.90)

df["geo_high_activity_day"] = (
    df["geo_hazard_event_count"].ge(threshold)
    & df["geo_available"]
).astype("int64")

df["hazard_day"] = (
    df["severe_weather_day"].eq(1)
    | df["geo_high_activity_day"].eq(1)
).astype(int)

# Build next-7-calendar-day labels only for complete calendar windows.
future_labels = pd.concat(
    [df["hazard_day"].shift(-day) for day in range(1, 8)],
    axis=1,
)

future_geo_available = pd.concat(
    [df["geo_available"].shift(-day) for day in range(1, 8)],
    axis=1,
)

next_dates = pd.concat(
    [df["date"].shift(-day) for day in range(1, 8)],
    axis=1,
)

calendar_complete = next_dates.apply(
    lambda row: (
        row.notna().all()
        and (row.iloc[0] - df.loc[row.name, "date"]).days == 1
        and all(
            (row.iloc[i] - row.iloc[i - 1]).days == 1
            for i in range(1, 7)
        )
    ),
    axis=1,
)

valid = (
    future_labels.notna().all(axis=1)
    & future_geo_available.fillna(False).all(axis=1)
    & calendar_complete
)

df["route_hazard_next_7_days"] = future_labels.max(axis=1)
df = df.loc[valid].copy()
df["route_hazard_next_7_days"] = (
    df["route_hazard_next_7_days"].astype(int)
)

df["label_source"] = "proxy_weather_plus_gdelt_high_activity"

df.drop(
    columns=[
        "severe_weather_day",
        "geo_high_activity_day",
        "hazard_day",
        "geo_available",
    ],
    inplace=True,
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Geo threshold:", round(threshold, 2))
print("Rows retained:", len(df))
print("Label distribution:")
print(df["route_hazard_next_7_days"].value_counts())
print("Date gaps:")
print(df["date"].diff().dt.days.value_counts().to_dict())
print("Saved:", OUTPUT_FILE)