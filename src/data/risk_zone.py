def create_risk_zones(route, n_zones=15):
    """
    Divide the full maritime route into geographic risk zones.

    The full route remains untouched.
    Zones are representative locations used later
    for weather and geopolitical monitoring.
    """

    coordinates = route.geometry.coordinates

    total_distance_km = route.properties["length"]
    total_duration_hours = route.properties["duration_hours"]

    if len(coordinates) <= n_zones:
        n_zones = len(coordinates)

    zones = []

    for i in range(n_zones):

        # Select evenly distributed route coordinates
        index = round(
            i * (len(coordinates) - 1) / (n_zones - 1)
        )

        longitude, latitude = coordinates[index]

        # Approximate position along the route
        progress = i / (n_zones - 1)

        distance_from_origin_km = (
            progress * total_distance_km
        )

        hours_from_origin = (
            progress * total_duration_hours
        )

        days_from_origin = hours_from_origin / 24

        zones.append({
            "zone_id": i + 1,
            "latitude": latitude,
            "longitude": longitude,
            "distance_from_origin_km": round(
                distance_from_origin_km, 2
            ),
            "hours_from_origin": round(
                hours_from_origin, 2
            ),
            "days_from_origin": round(
                days_from_origin, 2
            )
        })

    return zones


if __name__ == "__main__":

    import searoute as sr

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

    print("\n=== RISK ZONES ===")

    for zone in zones:

        print(
            f"Zone {zone['zone_id']:02d} | "
            f"Lat: {zone['latitude']:.4f} | "
            f"Lon: {zone['longitude']:.4f} | "
            f"{zone['distance_from_origin_km']:.0f} km | "
            f"{zone['days_from_origin']:.1f} days"
        )