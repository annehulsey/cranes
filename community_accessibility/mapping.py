from .base import *

def points_in_a_circle_polygon(center, r, n):
    [x, y, zone_n, zone_l] = utm.from_latlon(center[1], center[0])
    polygon = [None] * (n + 1)
    for theta in range(0, n + 1):
        x_i = x + np.cos(2 * np.pi / n * theta) * r
        y_i = y + np.sin(2 * np.pi / n * theta) * r
        polygon[theta] = utm.to_latlon(x_i, y_i, zone_n, zone_l)[::-1]

    return polygon


def cordon_geojson_generator(cordon_locations):
    df = cordon_locations.copy()

    # create a geojson for each cordon
    n = len(df)
    features = [None] * n
    for idx, i in zip(df.index.values, range(n)):
        center = (df.loc[idx, 'site.lng'], df.loc[idx, 'site.lat'])
        radius = df.loc[idx, 'radius']
        polygon = geojson.Polygon([points_in_a_circle_polygon(center, radius, n=32)])
        properties = {'id': idx}
        features[i] = geojson.Feature(geometry=polygon, properties=properties)

    cordon_geojson = geojson.FeatureCollection(features)

    return cordon_geojson


