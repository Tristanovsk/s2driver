import glob
import os
import numpy as np

import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from s2driver import driver_S2_SAFE as dS2

plt.ioff()
save = False

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen', 'orange', 'red'])
opj = os.path.join
imageSAFE = '/sat_data/satellite/sentinel2/L1C/31TFJ/S2A_MSIL1C_20201004T104031_N0209_R008_T31TFJ_20201004T125253.SAFE'
imageSAFE = '/sat_data/satellite/sentinel2/L1C/31TFJ/S2B_MSIL1C_20220731T103629_N0400_R008_T31TFJ_20220731T124834.SAFE'


l1c = dS2.s2image(imageSAFE,band_idx=[2,8],resolution=20)

# -----------------
# load angles
# -----------------
l1c.load_geom()


# -----------------
# load bands
# -----------------
l1c.load_bands()

# binary cmap
bcmap = ListedColormap(['khaki', 'lightblue'])

# Get geographic projection and bounds
extent = l1c.extent
proj = l1c.proj
bounds = extent.bounds
extent_val = [bounds.minx[0], bounds.maxx[0], bounds.miny[0], bounds.maxy[0]]

# Compute NDWI
green = l1c.bands.sel(bandID='B03')
nir = l1c.bands.sel(bandID='B8A')
ndwi = (green - nir) / (green + nir)


def water_mask(ndwi, threshold=0):
    water = xr.where(ndwi > threshold, 1, 0)
    return water.where(~np.isnan(ndwi))

coarsening = 1

# plot geometry
plt.figure()
axes = plt.subplot(1, 1, 1, projection=proj)
axes.set_extent(extent_val, proj)
fig = l1c.geom.razi.plot.imshow(col='bandID', cmap=cmap, col_wrap=4,extent=extent_val, transform=proj, aspect=1.15, robust=True)
for ax in fig.axes.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
# plt.savefig('./fig/example_scattering_angle_all_bands.png', dpi=300)
plt.show()

plt.figure(figsize=(20, 6))
shrink = 0.8
axes = plt.subplot(1, 3, 1, projection=proj)
axes.set_extent(extent_val, proj)
fig = ndwi[::coarsening, ::coarsening].plot.imshow(extent=extent_val, transform=proj, cmap=plt.cm.BrBG, robust=True,
                                   cbar_kwargs={'shrink': shrink})
# axes.coastlines(resolution='10m',linewidth=1)
axes.set_title('Sentinel 2, NDWI')

axes = plt.subplot(1, 3, 2, projection=proj)
water = water_mask(ndwi, 0.2)
axes.set_extent(extent_val, proj)
water[::coarsening, ::coarsening].plot.imshow(extent=extent_val, transform=proj, cmap=bcmap,
                              cbar_kwargs={'ticks': [0, 1], 'shrink': shrink})
axes.set_title('0.2 < NDWI')

axes = plt.subplot(1, 3, 3, projection=proj)
water = water_mask(ndwi, 0.)
axes.set_extent(extent_val, proj)
water[::coarsening, ::coarsening].plot.imshow(extent=extent_val, transform=proj, cmap=bcmap,
                              cbar_kwargs={'ticks': [0, 1], 'shrink': shrink})
axes.set_title('0. < NDWI')

#plt.savefig('./fig/example_ndwi_mask.png', dpi=300)

plt.show()


plt.figure()
fig = l1c.bands[:, ::10, ::10].plot.imshow(transform=ccrs.PlateCarree(),extent=extent_val,subplot_kws={'projection': proj, },
                                    col='bandID', col_wrap=4,  robust=True, cmap=plt.cm.binary_r)
plt.show()
for ax in fig.axes.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
#plt.savefig('./fig/example_reflectance_all_bands.png', dpi=300)

