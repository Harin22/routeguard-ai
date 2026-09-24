import os
import requests
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

API_KEY = os.getenv("NEWSDATA_API_KEY")

classifier = pipeline("sentiment-analysis")


def get_region(lat, lon):
    if 20 < lat < 40 and 30 < lon < 60:
        return "middle east"
    elif 40 < lat < 60 and -10 < lon < 40:
        return "europe"
    elif 5 < lat < 30 and 60 < lon < 100:
        return "south asia"
    else:
        return "global"


def fetch_news(region):
    url = "https://newsdata.io/api/1/latest"

    params = {
        "apikey": API_KEY,
        "q": f"{region} war OR conflict OR military",
        "language": "en"
    }

    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()

        articles = data.get("results", [])
        headlines = [
            a.get("title", "")
            for a in articles
            if a.get("title")
        ]

        if not headlines:
            raise Exception("No news")

        return headlines

    except:
        print(f"⚠️ API failed for {region}, using fallback")

        return [
            f"Conflict rising in {region}",
            f"Tension affecting {region} trade routes"
        ]


def analyze_conflict(news_list):
    scores = []

    for news in news_list[:5]:
        result = classifier(news)[0]

        if result["label"] == "NEGATIVE":
            scores.append(result["score"])
        else:
            scores.append(0)

    risk = sum(scores) / len(scores)

    return {
        "conflict_risk": round(risk, 3)
    }


def get_news_signal(lat, lon):
    region = get_region(lat, lon)
    news = fetch_news(region)

    return analyze_conflict(news)