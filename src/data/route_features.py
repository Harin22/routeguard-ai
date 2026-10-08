import searoute as sr


def build_route_features(start, end, n_zones=15):

    route = sr.searoute(
        start,
        end,
        units="km",
        append_orig_dest=True
    )

    total_distance = route.properties["length"]
    total_duration = route.properties["duration_hours"]
    total_duration_days = total_duration / 24

    coordinates = route.geometry.coordinates

    if len(coordinates) < n_zones:
        n_zones = len(coordinates)

    features = []

    for i in range(n_zones):

        index = round(
            i * (len(coordinates) - 1)
            / (n_zones - 1)
        )

        progress = i / (n_zones - 1)

        distance_from_origin = (
            progress * total_distance
        )

        days_from_origin = (
            progress * total_duration_days
        )

        features.append({
            "zone_id": i + 1,
            "route_distance_km": total_distance,
            "route_duration_hours": total_duration,
            "route_duration_days": total_duration_days,
            "zone_distance_from_origin_km": distance_from_origin,
            "zone_progress": progress,
            "zone_days_from_origin": days_from_origin
        })

    return features


if __name__ == "__main__":

    start = [
        -0.1278,
        51.5074
    ]

    end = [
        121.4737,
        31.2304
    ]

    features = build_route_features(
        start,
        end,
        n_zones=15
    )

    print("\nROUTE FEATURES")

    for zone in features:
        print(
            f"\nZone {zone['zone_id']:02d}"
        )

        print(
            f"Distance: "
            f"{zone['route_distance_km']:.2f} km"
        )

        print(
            f"Duration: "
            f"{zone['route_duration_hours']:.2f} hours"
        )

        print(
            f"Duration days: "
            f"{zone['route_duration_days']:.2f}"
        )

        print(
            f"Zone distance: "
            f"{zone['zone_distance_from_origin_km']:.2f} km"
        )

        print(
            f"Progress: "
            f"{zone['zone_progress']:.2f}"
        )

        print(
            f"Days from origin: "
            f"{zone['zone_days_from_origin']:.2f}"
        )