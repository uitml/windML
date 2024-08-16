import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader


if __name__ == __main__:
  df = pd.read_csv('data/meta.csv', index_col=0)

  stations_with_data = np.load('data/stations_with_data.npy', allow_pickle=True)
  df = df.loc[stations_with_data]

  shapefilename = shpreader.natural_earth(resolution='10m',
  category='cultural', name='admin_0_countries')

  reader = shpreader.Reader(shapefilename)
  countries = list(reader.records())

  for c in countries:
      if c.attributes['NAME_LONG'] == 'Norway':
        norway = c.geometry
        break

  projection = ccrs.Mercator()
  ax = plt.axes(projection=projection)
  ax.add_geometries([norway], projection, facecolor=(0.8, 0.8, 0.8))
  ax.plot(df.lon.values, df.lat.values, 'o')
  plt.show()