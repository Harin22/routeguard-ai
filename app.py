from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv

import logging
import time

#logging and monitoring

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("routeguard")


#Configuration and secrets

PROJECT_ROOT = Path(__file__).resolve().parent

# Local development: load .env if it exists.
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="RouteGuard AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Cloud: load secrets into environment variables.
# Existing local environment variables are preserved.
API_KEYS = (
    "NEWSDATA_API_KEY",
    "GEMINI_API_KEY",
    "MERCURY_API_KEY",
    "COHERE_API_KEY",
)

for key in API_KEYS:
    if not os.getenv(key):
        try:
            value = st.secrets.get(key)
            if value:
                os.environ[key] = str(value)
        except Exception:
            # Local .env remains usable without Streamlit secrets.
            pass

from src.data.ports import PORTS


# --------------------------------------------------
# Page styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }
    .rg-muted {
        color: #7c8798;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RouteGuard AI")
st.caption(
    "Maritime route risk intelligence · Weather · "
    "Geopolitical news · RAG assistant"
)
st.divider()


# --------------------------------------------------
# Port and route helpers
# --------------------------------------------------

PORT_OPTIONS = sorted(
    PORTS,
    key=lambda name: (PORTS[name]["country"], name),
)


def port_label(name: str) -> str:
    port = PORTS[name]
    return f"{name} — {port['country']} ({port['locode']})"


def get_coordinates(port_name: str) -> list[float]:
    port = PORTS[port_name]
    # SeaRoute expects [longitude, latitude].
    return [
        float(port["longitude"]),
        float(port["latitude"]),
    ]


def show_route_map(
    primary_route: dict,
    alternatives: list[dict] | None = None,
) -> None:
    alternatives = alternatives or []
    layers = []
    all_coords = primary_route.get("full_route", [])

    if all_coords:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[
                    {
                        "path": all_coords,
                        "route": "Primary route",
                    }
                ],
                get_path="path",
                get_color=[26, 111, 190],
                width_scale=15,
                width_min_pixels=3,
                pickable=True,
            )
        )

    alt_colors = (
        (220, 130, 45),
        (45, 155, 115),
    )

    for idx, alternative in enumerate(alternatives[:2]):
        coords = alternative.get("full_route", [])

        if coords:
            all_coords = all_coords + coords
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=[
                        {
                            "path": coords,
                            "route": f"Alternative {idx + 1}",
                        }
                    ],
                    get_path="path",
                    get_color=list(alt_colors[idx]),
                    width_scale=12,
                    width_min_pixels=2,
                    pickable=True,
                )
            )

    if not all_coords or not layers:
        st.info("Route geometry is not available to display.")
        return

    # Basic viewport; long routes may cross the dateline.
    center_lon = (
        sum(float(point[0]) for point in all_coords)
        / len(all_coords)
    )
    center_lat = (
        sum(float(point[1]) for point in all_coords)
        / len(all_coords)
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=1,
            pitch=0,
        ),
        tooltip={"text": "{route}"},
        map_style=(
            "https://basemaps.cartocdn.com/gl/"
            "positron-gl-style/style.json"
        ),
    )

    st.pydeck_chart(deck, width="stretch")


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_route_label" not in st.session_state:
    st.session_state.analysis_route_label = None

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []


# --------------------------------------------------
# Route planner
# --------------------------------------------------

with st.container(border=True):
    st.subheader("Route planner")
    st.caption(
        f"Select from the {len(PORTS)} configured ports. "
        "Each entry includes country, UN/LOCODE and coordinates."
    )

    from_col, to_col = st.columns(2)

    with from_col:
        origin = st.selectbox(
            "From",
            options=PORT_OPTIONS,
            index=None,
            placeholder="Choose departure port",
            format_func=port_label,
            key="origin_port",
        )

    with to_col:
        destination = st.selectbox(
            "To",
            options=PORT_OPTIONS,
            index=None,
            placeholder="Choose destination port",
            format_func=port_label,
            key="destination_port",
        )

    run_analysis = st.button(
        "Start Route Analysis",
        type="primary",
        width="stretch",
    )


#Run route analysis + monitoring

if run_analysis:
    if not origin or not destination:
        st.warning("Select both a departure port and a destination.")

    elif origin == destination:
        st.warning("Choose two different ports.")

    else:
        start_time = time.perf_counter()

        logger.info(
            "Route analysis started: %s -> %s",
            origin,
            destination,
        )

        try:
            from src.data.route_selection import select_route

            with st.spinner(
                "Fetching fresh news across route zones, updating "
                "ChromaDB, analyzing geopolitics, and fetching "
                "weather. This can take a few minutes..."
            ):
                result = select_route(
                    get_coordinates(origin),
                    get_coordinates(destination),
                )

            elapsed = time.perf_counter() - start_time
            risk_score = float(result["final_risk_score"])

            logger.info(
                "Route analysis completed | %s -> %s | "
                "Duration: %.2f seconds | Risk: %.2f/10",
                origin,
                destination,
                elapsed,
                risk_score,
            )

            st.session_state.analysis_result = result
            st.session_state.analysis_route_label = (
                f"{origin} → {destination}"
            )
            st.session_state.rag_messages = []

        except Exception:
            elapsed = time.perf_counter() - start_time

            logger.exception(
                "Route analysis failed | %s -> %s | "
                "Duration before failure: %.2f seconds",
                origin,
                destination,
                elapsed,
            )

            st.error(
                "Route analysis could not complete. Check API "
                "credentials, network access, and the application logs."
            )


    #Weather table
    st.subheader("Weather along the route")
    weather_details = result.get("weather_zone_details", [])

    if weather_details:
        weather_df = pd.DataFrame(weather_details)

        display_columns = {
            "zone_id": "Zone",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "forecast_time_utc": "Forecast time (UTC)",
            "temperature_2m": "Avg temperature (°C)",
            "precipitation": "Precipitation (mm / 24h)",
            "rain": "Rain (mm / 24h)",
            "pressure_msl": "Avg pressure (hPa)",
            "cloud_cover": "Avg cloud cover (%)",
            "wind_speed_10m": "Avg wind (kn)",
            "wind_gusts_10m": "Avg gust (kn)",
            "wind_direction_10m": "Avg wind direction (°)",
            "weather_code_now": "Current weather code",
        }

        columns = [
            column
            for column in display_columns
            if column in weather_df.columns
        ]

        st.dataframe(
            weather_df[columns]
            .rename(columns=display_columns)
            .round(2),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Weather summaries use the first 24 forecast hours at "
            "each monitoring zone. Precipitation/rain are totals; "
            "other numeric fields are averages."
        )

    else:
        st.info(
            "Weather aggregate scores were calculated, but per-zone "
            "details are not yet being returned by the current "
            "route-selection module."
        )

    #Alternative routes

    # Alternative routes
    st.subheader("Alternative routes")

    alternatives = result.get("alternatives", [])

    if alternatives:
        for alternative in alternatives:
            with st.container(border=True):
                st.markdown(
                    f"**Alternative {alternative['option']}**"
                )

                a1, a2 = st.columns(2)

                a1.metric(
                    "Distance",
                    f"{float(alternative['distance_km']):,.0f} km",
                )

                a2.metric(
                    "Estimated duration",
                    f"{float(alternative['duration_hours']):,.1f} hours",
                )

                restrictions = alternative.get("restrictions", [])

                st.caption(
                    "Restrictions: "
                    f"{', '.join(map(str, restrictions)) or 'Default'}"
                )

        st.warning(
            "These candidates are currently generated and sorted "
            "by distance. The existing code does not independently "
            "calculate a risk score for each alternative yet."
        )
    else:
        st.info(
            "The primary route is below the configured threshold, "
            "so the current route-selection logic has not generated "
            "alternatives."
        )

    # Retrieved evidence
    st.subheader("Retrieved geopolitical evidence")

    evidence = result.get("geopolitical_evidence", [])

    if evidence:
        for zone in evidence:
            with st.expander(
                f"Zone {zone.get('zone_id', '?')} · "
                f"{zone.get('latitude', 0):.3f}, "
                f"{zone.get('longitude', 0):.3f}"
            ):
                documents = zone.get("documents", [])

                if not documents:
                    st.write("No documents retrieved for this zone.")

                for index, document in enumerate(documents, start=1):
                    st.markdown(f"**Retrieved item {index}**")
                    st.write(document)
    else:
        st.info(
            "No route-level evidence list was returned by the "
            "current route-selection module."
        )


    # --------------------------------------------------
    # RAG chat
    # --------------------------------------------------

    st.divider()
    st.header("Ask RouteGuard about the news")
    st.caption(
        "Ask questions about retrieved maritime news. Answers "
        "depend on the documents available in the ChromaDB collection."
    )

    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        title = source.get("title", "Untitled source")
                        url = source.get("url")

                        if url:
                            st.markdown(
                                f"- [{title}]({url}) · "
                                f"{source.get('source', 'Unknown source')} · "
                                f"{source.get('published', 'Unknown date')}"
                            )
                        else:
                            st.markdown(
                                f"- {title} · "
                                f"{source.get('source', 'Unknown source')} · "
                                f"{source.get('published', 'Unknown date')}"
                            )

    question = st.chat_input(
        "Ask about news or risks affecting this route…"
    )

    if question:
        history = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in st.session_state.rag_messages
        ]

        st.session_state.rag_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        try:
            from src.rag.chat_service import chat_with_routeguard

            with st.spinner(
                "Retrieving news and drafting an evidence-grounded answer…"
            ):
                answer = chat_with_routeguard(
                    question,
                    history=history,
                    top_k=5,
                    route_key=result.get("route_key"),
                )

            st.session_state.rag_messages.append(
                {
                    "role": "assistant",
                    "content": answer["answer"],
                    "sources": answer.get("sources", []),
                    "provider": answer.get("provider", "Unknown"),
                }
            )

        except Exception as exc:
            st.session_state.rag_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "The RAG assistant could not answer this query. "
                        "Check the API credentials and ChromaDB collection. "
                        f"Error: {exc}"
                    ),
                    "sources": [],
                }
            )

        st.rerun()

else:
    st.info(
        "Choose an origin and destination above, then select "
        "**Start Route Analysis**. Results appear here after "
        "analysis completes."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption(
    "RouteGuard AI is an experimental decision-support system. "
    "It does not replace official navigational, weather, security, "
    "or port authority advisories."
)