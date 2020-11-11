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


def map_buildings(bldgs, attribute, color_breaks):
    # prep the dataframe for linking to the vector tile
    df = bldgs.copy()
    join_dim = 'match_id'
    df[join_dim] = [x[2:] for x in list(df.index.values)]

    # convert height to meters for extrusion
    height = 'building.building_ht_ft'
    extrusion = 'hgt_m'
    df[extrusion] = df[height] / 3.28
    height_stops = [[0, 0], [1000, 1000]]

    # prep the attributes for readability
    if attribute == 'building.building_type_id':
        df[attribute] = [x[:-1] for x in list(df[attribute].values)]

    # create simplified dataset
    columns = [join_dim, attribute, extrusion]
    data = df.loc[:, columns]

    # prep the color breaks
    if color_breaks == 'match':
        colors = 'Paired'
        color_breaks = np.unique(df[attribute])
        color_stops = create_color_stops(color_breaks, colors=colors)
        color_function_type = 'match'
    elif len(color_breaks) > 1:
        colors = 'YlGnBu'
        color_stops = create_color_stops(color_breaks, colors=colors)
        color_function_type = 'interpolate'
    else:
        n = color_breaks[0]
        width = int(100 / n)
        sig_digits = 0
        colors = 'YlGnBu'
        color_breaks = [round(data[attribute].quantile(q=x * 0.01), sig_digits) for x in range(width, 101, width)]
        # categorize linearly if quantiles produce zeros
        if color_breaks[1] == 0:
            color_breaks = np.round(np.linspace(color_breaks[0], color_breaks[-1], n, endpoint=True), sig_digits)
        color_stops = create_color_stops(color_breaks, colors=colors)
        color_function_type = 'interpolate'

    data = json.loads(data.to_json(orient='records'))

    viz = ChoroplethViz(data,
                        access_token=token,
                        vector_url='mapbox://ahulsey.0ugpotnv',
                        vector_layer_name='SF_Empty_Footprints-8gvb25',
                        vector_join_property='sf16_BldgI',
                        data_join_property=join_dim,
                        color_property=attribute,
                        color_stops=color_stops,
                        color_function_type=color_function_type,
                        color_default='gray',
                        height_property=extrusion,
                        height_stops=height_stops,
                        height_function_type='interpolate',
                        opacity=1,
                        center=(-122.4, 37.787),
                        zoom=14,
                        bearing=-15,
                        pitch=45,
                        legend_layout='vertical',
                        legend_key_shape='bar',
                        legend_key_borders_on=False)
    viz.show()