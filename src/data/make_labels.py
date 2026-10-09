
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

df = pd.read_csv(
    ROOT / "data/historical_weather_daily.csv",
    parse_dates=["date"],
).sort_values("date").reset_index(drop=True)

STORM_CODES = {65, 75, 82, 86, 95, 96, 99}

# Define a severe-weather day
df["severe_day"] = (
    (df["wind_gusts_10m"] >= 60)
    | df["weather_code"].isin(STORM_CODES)
).astype(int)

# Target: severe weather during the following 7 days
future = pd.concat(
    [df["severe_day"].shift(-day) for day in range(1, 8)],
    axis=1,
)

complete = future.notna().all(axis=1)
df["severe_weather_next_7_days"] = future.max(axis=1)

# Remove incomplete windows and intermediate label
df = df.loc[complete].copy()
df["label_source"] = "proxy_weather"
df.drop(columns=["severe_day"], inplace=True)

output = ROOT / "data/processed/training_dataset.csv"
output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False)

print(f"Rows saved: {len(df)}")
print(f"Positive labels: {int(df['severe_weather_next_7_days'].sum())}")
print(f"Positive rate: {df['severe_weather_next_7_days'].mean():.1%}")
print(f"Saved to: {output}")
