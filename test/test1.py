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

from s2driver import driver_S2_SAFE as S2


imageSAFE_v3 = '/sat_data/satellite/sentinel2/L1C/31TFJ/S2A_MSIL1C_20201004T104031_N0209_R008_T31TFJ_20201004T125253.SAFE'
imageSAFE_v4 = '/sat_data/satellite/sentinel2/L1C/31TFJ/S2B_MSIL1C_20220731T103629_N0400_R008_T31TFJ_20220731T124834.SAFE'

bandIds = [0,1,2,3,8,12]
band_names = S2.INFO[bandIds]
resolution=20
ESAbands = band_names.loc['ESA']

image = imageSAFE_v4
l1c = S2.s2image(image,band_idx=bandIds,resolution=resolution)

l1c.load_bands()


from affine import Affine
opj = os.path.join
satdir = '/sat_data/satellite/sentinel2/L2A/GRS/31TGM'

image = 'S2*_v14.nc'
files = glob.glob(opj(satdir, image))
product = xr.open_dataset(files[0], chunks={'x': 512, 'y': 512}, decode_coords="all",engine='netcdf4')
i2m = product.crs.i2m
wkt = product.crs.wkt
#set geotransform
i2m = np.array((product.crs.i2m.split(','))).astype(float)
gt = Affine(i2m[0], i2m[1], i2m[4], i2m[2], i2m[3],i2m[5])
product.rio.write_transform(gt, inplace=True)
product.rio.write_crs(32631,inplace=True)
gt = product.rio.transform()
x_,y_ = product.x,product.y
nx,ny=len(x_),len(y_)
res=20
x0,y0 = gt*(y_+0.5,x_+0.5)
product['x']= x0[:,0].values
product['y']= y0[0,:].values