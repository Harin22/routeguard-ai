def generate_route_points(start, end, n_points=10):
    start_lat, start_lon = start
    end_lat, end_lon = end

    points = []

    for i in range(n_points + 1):
        t = i / n_points  

        lat = start_lat + t * (end_lat - start_lat)
        lon = start_lon + t * (end_lon - start_lon)

        points.append((lat, lon))

    return points

'''predicting point'''

if __name__ == "__main__":
    start = (31.23, 121.47)   # Shanghai
    end = (51.92, 4.48)       # Rotterdam

    route = generate_route_points(start, end, n_points=5)

    for p in route:
        print(p)