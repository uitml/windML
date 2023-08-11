from functools import partial

import pandas as pd


from bokeh.layouts import column, row, layout
from bokeh.plotting import figure, curdoc
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models import ColumnDataSource, Spinner, CheckboxButtonGroup, Button, Dropdown, Div

import numpy as np


def wgs84_to_web_mercator(df, lon="lon", lat="lat"):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi / 180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360.0)) * k
    return df


def load_data(path: str = './', mercator_proj: bool = True):
    station_meta = pd.read_csv(f'{path}meta.csv', index_col=0)
    feature_data = pd.read_csv(f'{path}raw_data.csv', index_col=(0,1))
    feature_data.index = feature_data.index.set_levels(pd.to_datetime(feature_data.index.levels[1]), level=1)

    feature_meta = pd.read_csv(f'{path}feature_meta.csv', index_col=(0,1,2))
    feature_meta = feature_meta.rename_axis(["station", "time", 'elementId'])
    feature_meta.index = feature_meta.index.set_levels(pd.to_datetime(feature_meta.index.levels[1]), level=1)


    station_meta = station_meta.loc[feature_data.index.get_level_values(0).unique()]


    if mercator_proj:
        station_meta = wgs84_to_web_mercator(station_meta)

    collection_dict = {'station_meta': station_meta,
                       'feature_data': feature_data,
                       'feature_meta': feature_meta
                       }

    return collection_dict


def generate_map(tooltip=None):
    tile_provider = get_provider(Vendors.CARTODBPOSITRON)

    if tooltip is not None:
        map_figure = figure(x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
                        x_axis_type="mercator",
                        y_axis_type="mercator", width=1000, height=1000,
                        tooltips=tooltip)
    # range bounds supplied in web mercator coordinates
    else:
        map_figure = figure(x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
                            x_axis_type="mercator",
                            y_axis_type="mercator") # , sizing_mode='scale_both')

    map_figure.add_tile(tile_provider)

    return map_figure


def dropdown_response(event, element):
    element.label = event.item


def filter_stations(df, feature, dropdowns):
    start = True
    df_filter = None
    for dropdown in dropdowns:
        colname = dropdown.children[0].text
        target = dropdown.children[1].label

        if target == 'all':
            continue

        if start:
            df_filter = df.loc[:, :, feature][colname] == target
            start = False
            continue

        df_filter = (df_filter) & (df.loc[:, :, feature][colname] == target)

    if df_filter is None:
        return df.loc[:, :, feature]

    res_df = df.loc[:, :, feature][df_filter]
    if res_df.empty:
        return None

    return df.loc[:, :, feature][df_filter]


def color_stations(df, feature):

    if feature == 'all':
        return ['magenta' for _ in range(len(df.index.get_level_values(0).unique()))]


    if type(df[feature][0]) == str:
        values = np.array([ord(char) - 65 for char in df[feature]])

    else:
        values = df[feature].values

    n_colors = values.max()
    print('FEATURE:::: ', feature, n_colors)
    colors = np.array([((1 / n_colors) * i, 1 - (1 / n_colors) * i, 0) for i in range(n_colors+1)])

    return colors[values]


def label_stations(df, feature):
    if feature == 'all':
        return ['unfiltered' for _ in range(len(df.index.get_level_values(0).unique()))]

    labels = df[feature].values.astype(str)

    return list(np.char.add(f'{feature}: ', labels))


def run_filter(dropdowns, data, sources):
    print("\n----------\n\n")

    data_feature = dropdowns[0].children[1].label
    color_by = dropdowns[1].children[1].label

    filtered_feature_meta = filter_stations(data['feature_meta'], data_feature, dropdowns[2:])

    if filtered_feature_meta is None:
        temp_data = dict(lat=[],
                     lon=[],
                     ids=[],
                     colors=[],
                     #label=labels[:len(meta.values)]
                     )
        sources.data = temp_data
        return

    stations = filtered_feature_meta.index.get_level_values(0).unique()

    station_meta = data['station_meta'].loc[stations]

    #labels = label_stations(filtered_feature_meta, color_by)

    colors = color_stations(filtered_feature_meta, color_by)

    temp_data = dict(lat=station_meta['y'].values,
                     lon=station_meta['x'].values,
                     ids=station_meta.index,
                     colors=colors,
                     #label=labels
                     )

    sources.data = temp_data


def add_dropdown(df, feature: str = None, include_all: bool = False):

    if feature is None:
        menu = df.columns.to_list()
        text= 'Features'
    else:
        menu = list(df[feature].unique())
        text=feature

    if include_all:
        label = 'all'
        menu.insert(0, label)
    else:
        label = menu[0]


    dropdown = Dropdown(
                        label=label,
                        menu=menu,
                        width=200
                        )
    dropdown.on_click(partial(dropdown_response, element=dropdown))

    title = Div(text=f'{text}',
                width=200)
    return column([title, dropdown])


if __name__ == '__main__':
    data = load_data('data/')
    print(data['feature_data'].columns)
    print(data['feature_meta'].columns)

    filter_stations(data['feature_meta'], 'wind_speed_of_gust', 'qualityCode', 0)


if 'bokeh_app' in __name__:
    data_dict = load_data('data/')

    sources = ColumnDataSource(data=dict(lat=data_dict['station_meta']['y'].values,
                                         lon=data_dict['station_meta']['x'].values,
                                         ids=data_dict['station_meta'].index.values,
                                         colors=['magenta' for _ in range(len(data_dict['station_meta']))],
                                         labels=['unfiltered' for _ in range(len(data_dict['station_meta']))]
                                         ))
    tooltip = [
        ('ID: ', "@ids")
    ]
    map_figure = generate_map(tooltip)

    stations = map_figure.circle('lon', 'lat', source=sources, color='colors', size=2)


    source_scaler = Spinner(
        title='Station size',
        low=2,
        high=10,
        value=stations.glyph.size,
        width=200
        )
    source_scaler.js_link('value', stations.glyph, 'size')
    '''
    dropdown = Dropdown(label=data_dict['feature_data'].columns[0],
                        menu=data_dict['feature_data'].columns.to_list(),
                        width=200)
    dropdown.on_click(partial(dropdown_response, element=dropdown))
    '''

    print(data_dict['feature_meta'].columns)

    dropdown_list = []
    dropdown_list.append(add_dropdown(data_dict['feature_data']))
    dropdown_list.append(add_dropdown(data_dict['feature_meta'][['performanceCategory', 'qualityCode', 'exposureCategory']], include_all=True))
    dropdown_list[1].children[0].text = 'Color by'

    dropdown_list.append(add_dropdown(data_dict['feature_meta'], 'timeResolution', True))
    dropdown_list.append(add_dropdown(data_dict['feature_meta'], 'timeOffset', True))



    print("\n\n\n")
    print(dropdown_list[1:])
    print("\n\n\n")
    button = Button(label="run", button_type="success")
    button.on_click(partial(run_filter, dropdowns=dropdown_list, data=data_dict, sources=sources))

    layout_list = [[source_scaler, dropdown_list[0], dropdown_list[1], button],
                   dropdown_list[2:],
                   [map_figure]
                   ]

    curdoc().add_root(layout(layout_list))
