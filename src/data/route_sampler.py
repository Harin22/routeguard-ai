import searoute as sr


def generate_maritime_route(start, end):
    """
    Generate the shortest maritime route between two coordinates.

    Coordinates must be:
        [longitude, latitude]
    """

    route = sr.searoute(
        start,
        end,
        units="km",
        append_orig_dest=True
    )

    return route


def sample_route_points(route, n_points=5):
    """
    Select evenly distributed points along the full route.

    The complete route geometry is preserved.
    These points are only representative monitoring points.
    """

    coordinates = route.geometry.coordinates

    if len(coordinates) <= n_points:
        return coordinates

    indices = [
        round(i * (len(coordinates) - 1) / (n_points - 1))
        for i in range(n_points)
    ]

    sampled_points = [
        coordinates[index]
        for index in indices
    ]

    return sampled_points


def build_route(start, end, n_points=5):
    """
    Generate a maritime route and its monitoring points.
    """

    route = generate_maritime_route(start, end)

    full_route = route.geometry.coordinates

    monitoring_points = sample_route_points(
        route,
        n_points=n_points
    )

    return {
        "distance_km": route.properties["length"],
        "duration_hours": route.properties["duration_hours"],
        "full_route": full_route,
        "monitoring_points": monitoring_points
    }


if __name__ == "__main__":

    #Shanghai → Rotterdam
    start = [121.4737, 31.2304]
    end = [4.4777, 51.9244]

    result = build_route(
        start,
        end,
        n_points=5
    )

    print("\n=== ROUTE INFORMATION ===")

    print(
        f"Distance: "
        f"{result['distance_km']:.2f} km"
    )

    print(
        f"Duration: "
        f"{result['duration_hours']:.2f} hours"
    )

    print(
        f"Full route coordinates: "
        f"{len(result['full_route'])}"
    )

    print("\n=== monitoring point ===")

    for i, point in enumerate(
        result["monitoring_points"],
        start=1
    ):
        longitude, latitude = point

        print(
            f"P{i}: "
            f"Latitude={latitude:.4f}, "
            f"Longitude={longitude:.4f}"
        )