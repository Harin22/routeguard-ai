
import searoute as sr


def generate_maritime_route(start, end, restrictions=None):
    """
    Generate a maritime route between two coordinates.

    Coordinates: [longitude, latitude]
    """
    kwargs = {
        "units": "km",
        "append_orig_dest": True,
    }

    if restrictions is not None:
        kwargs["restrictions"] = restrictions

    return sr.searoute(start, end, **kwargs)


def sample_route_points(route, n_points=5):
    """
    Select evenly distributed monitoring points along a route.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    coordinates = route.geometry.coordinates

    if len(coordinates) <= n_points:
        return coordinates

    indices = [
        round(i * (len(coordinates) - 1) / (n_points - 1))
        for i in range(n_points)
    ]

    return [coordinates[index] for index in indices]


def build_route(start, end, n_points=5):
    """
    Build one sea route with distance, duration and monitoring points.
    """
    route = generate_maritime_route(start, end)

    return {
        "distance_km": route.properties["length"],
        "duration_hours": route.properties["duration_hours"],
        "full_route": route.geometry.coordinates,
        "monitoring_points": sample_route_points(route, n_points),
    }


def generate_alternative_routes(start, end, n_points=5):
    """
    Generate distinct route candidates using different passage restrictions.

    Candidates are deduplicated by route geometry and sorted by distance.
    """
    configurations = [
        ["northwest"],
        ["northwest", "suez"],
        ["northwest", "gibraltar"],
        ["northwest", "suez", "gibraltar"],
    ]

    candidates = []
    seen_geometries = set()

    for restrictions in configurations:
        try:
            route = generate_maritime_route(
                start,
                end,
                restrictions=restrictions,
            )

            coordinates = route.geometry.coordinates

            if not coordinates:
                continue

            geometry_key = tuple(
                (round(lon, 4), round(lat, 4))
                for lon, lat in coordinates
            )

            if geometry_key in seen_geometries:
                continue

            seen_geometries.add(geometry_key)

            candidates.append({
                "restrictions": restrictions,
                "distance_km": route.properties["length"],
                "duration_hours": route.properties["duration_hours"],
                "full_route": coordinates,
                "monitoring_points": sample_route_points(
                    route, n_points=n_points
                ),
            })

        except Exception as error:
            print(
                f"Skipping restrictions {restrictions}: {error}"
            )

    candidates.sort(key=lambda item: item["distance_km"])
    return candidates


if __name__ == "__main__":
    # Shanghai -> Rotterdam
    start = [121.4737, 31.2304]
    end = [4.4777, 51.9244]

    result = build_route(start, end, n_points=5)

    print("\n=== DEFAULT SEA ROUTE ===")
    print(f"Distance: {result['distance_km']:.2f} km")
    print(f"Duration: {result['duration_hours']:.2f} hours")
    print(f"Geometry points: {len(result['full_route'])}")

    print("\n=== DISTINCT SEA-ROUTE CANDIDATES ===")
    alternatives = generate_alternative_routes(
        start, end, n_points=5
    )

    for index, route in enumerate(alternatives, start=1):
        print(
            f"Route {index}: "
            f"{route['distance_km']:.2f} km | "
            f"{route['duration_hours']:.2f} hours | "
            f"Restrictions: {route['restrictions']}"
        )
