
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
# import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# mpl.use('TkAgg')
from osgeo import gdal, ogr
import scipy.odr as odr

# import rasterio as rio

import eoreader as eo
from eoreader.reader import Reader
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen', 'orange', 'red'])
opj = os.path.join
imageSAFE = '/sat_data/satellite/sentinel2/L1C/31TFJ/S2A_MSIL1C_20201004T104031_N0209_R008_T31TFJ_20201004T125253.SAFE/'
abspath=os.path.abspath(imageSAFE)
dirroot,basename = os.path.split(abspath)

xml_granule = glob.glob(opj(imageSAFE, 'GRANULE', '*', 'MTD_TL.xml'))[0]
xml_file = glob.glob(opj(imageSAFE, 'MTD*.xml'))[0]


band_names = pd.DataFrame(dict(band=('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12')))
band_id =band_names.band.str.replace('B','')
band_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

norm = mpl.colors.Normalize(vmin=0, vmax=11)
reader =Reader()
# Open the product
prod = reader.open(imageSAFE, remove_tmp=True)
processing_baseline = prod._processing_baseline
if processing_baseline < 4:
    mask_names = eo.products.optical.s2_product.S2GmlMasks.__members__.keys()

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.1, top=0.8, left=0.1, right=0.8)
    w_=500
    xpt = 672000.0
    xpt = 646800.0

    axins = zoomed_inset_axes(ax, 45, loc='upper right', borderpad=-7)
    for i,id in enumerate(band_id):
        detfoos = prod._open_mask_lt_4_0('DETFOO',id)
        detfoos['name']=detfoos.gml_id.str.replace('detector_footprint-','')
        detfoos.boundary.plot(ax=ax,color=cmap(norm(i)),label='B'+id)

        detfoos.boundary.plot(ax=axins,color=cmap(norm(i)))

        axins.set_xlim(xpt-w_,xpt+w_)
        axins.set_ylim(4.857e6,4.858e6)

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    axins.set(xticks=[], yticks=[])
    ax.legend(loc='lower right',bbox_to_anchor=(1, 0))
    plt.savefig('./fig/example_all_detectors_v3.png', dpi=300)


    fig, axs = plt.subplots(3,4,figsize=(15, 15))
    axs=axs.ravel()
    for i,id in enumerate(band_id):
        detfoos = prod._open_mask_lt_4_0('DETFOO',id)
        detfoos['name']=detfoos.gml_id.str.replace('detector_footprint-','')
        ax=axs[i]
        detfoos.plot(ax=ax,column='name',legend=True)
        ax.get_legend().set_bbox_to_anchor((1,1))
        ax.get_legend().set_title("Detector footprint")
    axs[-1].set_visible(False)
    plt.tight_layout()
    prod._open_clouds_lt_4_0()

else:
    prod._open_mask_gt_4_0('DETFOO','08')


prod._open_clouds_lt_4_0('ALL_CLOUDS')



ds = gdal.Open(xml_file)
metadata = ds.GetMetadata()
subds = ds.GetMetadata('SUBDATASETS')

for i in range(4):
    gdal.ErrorReset()
    ds = gdal.Open(subds['SUBDATASET_%d_NAME' % (i + 1)])
    assert ds is not None and gdal.GetLastErrorMsg() == '', \
        subds['SUBDATASET_%d_NAME' % (i + 1)]


##############################################
# Internal parsing function for angular grids
def parse_angular_grid_node(node):
    values = []
    for c in node.find('Values_List'):
        values.append(np.array([float(t) for t in c.text.split()]))
        values_array = np.stack(values)
    return values_array


with open(xml_granule) as xml_file:
    tree = ET.parse(xml_file)
    root = tree.getroot()

sza = parse_angular_grid_node(root.find('.//Tile_Angles/Sun_Angles_Grid/Zenith'))
sazi = parse_angular_grid_node(root.find('.//Tile_Angles/Sun_Angles_Grid/Azimuth'))

# check dimension
Nx, Ny = sza.shape

sun_ang = xr.Dataset(data_vars=dict(sza=(['x', 'y'], sza),
                                    sazi=(['x', 'y'], sazi)),
                     coords=dict(x=range(Nx), y=range(Ny)))

# ---------------------------------
# getting viewing geometry datacube
# ---------------------------------

bandIds, detectorIds = [], []
for angleID in root.findall('.//Tile_Angles/Viewing_Incidence_Angles_Grids'):
    bandIds.append(int(angleID.attrib['bandId']))
    detectorIds.append(int(angleID.attrib['detectorId']))
Nband, Ndetector = np.max(bandIds) + 1, np.max(bandIds) + 1

# allocate/fill rasters
vza, vazi = np.full((Nband, Ndetector, Nx, Ny), np.nan, dtype=float), np.full((Nband, Ndetector, Nx, Ny), np.nan,
                                                                              dtype=float)

for angleID in root.findall('.//Tile_Angles/Viewing_Incidence_Angles_Grids'):
    iband = int(angleID.attrib['bandId'])
    idetector = int(angleID.attrib['detectorId'])
    vza[iband, idetector] = parse_angular_grid_node(angleID.find('Zenith'))
    vazi[iband, idetector] = parse_angular_grid_node(angleID.find('Azimuth'))

view_ang = xr.Dataset(data_vars=dict(vza=(['bandId', 'detectorId', 'x', 'y'], vza),
                                     vazi=(['bandId', 'detectorId', 'x', 'y'], vazi)),
                      coords=dict(bandId=range(Nband),
                                  detectorId=range(Ndetector),
                                  x=range(Nx), y=range(Ny)))
# clean up Dataset (remove empty slices)
view_ang = view_ang.dropna('detectorId', how='all')


# ---------------------------------
# getting detector masks
# ---------------------------------
mask_ = []
for maskID in root.findall('.//Pixel_Level_QI/MASK_FILENAME'):
    type = maskID.attrib['type']
    if type != 'MSK_CLOUDS':
        iband = 0
        iband = int(maskID.attrib['bandId'])
    mask_.append([type,iband,maskID.text])
    print(type)
maskinfo = pd.DataFrame(mask_,columns=['type','iband','file']).set_index(['type','iband'])#.to_xarray()
gml =maskinfo.loc['MSK_DEFECT',0].file # maskinfo.sel(type='MSK_DETFOO',iband=0).file.values
from lxml.etree import parse
maskfile = opj(abspath,gml)
mask = ogr.Open(maskfile)
parse(maskfile).getroot()
gpd.read_file(maskfile)
gpd.read_file(maskfile, driver='GML')


# -------------------------
# interpolation
# -------------------------
# tile for 10m resolution: width,height = 10980,10980
# tile for 20m resolution: width,height = 5490,5490
# tile for 60m resolution: width,height = 1830,1830

width, height = 5490, 5490
new_x = np.linspace(0, Nx, width)
new_y = np.linspace(0, Ny, height)
# -------------------------
# Sun angles (easy!)

method = 'linear'
new_sun_ang = sun_ang.interp(x=new_x, y=new_y, method=method)
new_sun_ang = new_sun_ang.assign_coords(x=range(width), y=range(height))
# fig,axs=plt.subplots(1,2,figsize=(10,5))
# new_sun_ang.sza.plot(ax=axs[0],cmap=plt.cm.Spectral_r)
# new_sun_ang.sazi.plot(ax=axs[1],cmap=plt.cm.Spectral_r)

# -------------------------
# Viewing angles (not easy!)

view_ang.vza.isel(bandId=0).plot.imshow(col='detectorId', col_wrap=3, origin='upper', cmap=cmap)
view_ang.vazi.isel(bandId=0).plot.imshow(col='detectorId', col_wrap=3, origin='upper', cmap=cmap)

dx = np.tan(np.deg2rad(view_ang.vza)) * np.sin(np.deg2rad(view_ang.vazi))
dy = np.tan(np.deg2rad(view_ang.vza)) * np.cos(np.deg2rad(view_ang.vazi))

dx.isel(bandId=0).plot.imshow(col='detectorId', col_wrap=3, origin='upper', cmap=cmap)
dy.isel(bandId=0).plot.imshow(col='detectorId', col_wrap=3, origin='upper', cmap=cmap)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.985, left=0.1, right=0.98,
                    hspace=0.01, wspace=0.025)

for id in range(5):
    dx.isel(bandId=0, detectorId=id).plot(ax=axs[0], cmap=cmap, vmin=dx.min(), vmax=dx.max())
    dy.isel(bandId=0, detectorId=id).plot(ax=axs[1], cmap=cmap, vmin=dy.min(), vmax=dy.max())

raw_res = 5000
res = 200


# ---------------------------------
# test with ODR multilinear regression
# ---------------------------------

def linfit(beta, x):
    return beta[0] * x[0] + beta[1] * x[1] + beta[2]

def data_fitting(x0, y0, arr, verbose=False):
    xgrid, ygrid = np.meshgrid(x0, y0, indexing='ij')

    # vectorize
    values = arr.values.flatten()
    x_ = xgrid.flatten()
    y_ = ygrid.flatten()

    # remove NaN
    idx = ~np.isnan(values)
    values = values[idx]
    points = np.empty((2, len(values)))
    points[0] = x_[idx]
    points[1] = y_[idx]

    # set ODR fitting
    mean = np.nanmean(values)
    linear = odr.Model(linfit)
    data = odr.Data(points, values)
    beta0 = [0, 0, mean]

    # proceed with ODR fitting
    fit = odr.ODR(data, linear, beta0=beta0)
    resfit = fit.run()

    if verbose:
        resfit.pprint()

    return resfit.beta



xarr = dy

kwarg = dict(vmin=xarr.min(), vmax=xarr.max())
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(22, 8), sharex=True, sharey=True)
fig.subplots_adjust(bottom=0.05, top=0.965, left=0.05, right=0.98,
                    hspace=0.12, wspace=0.1)
for id in range(5):

    arr = xarr.isel(bandId=0, detectorId=id).dropna('y', how='all').dropna('x', how='all')
    x0, y0 = arr.x.values, arr.y.values

    beta = data_fitting(x0, y0, arr)

    xnew = np.linspace(x0.min()-1, x0.max()+1, 500)
    ynew = np.linspace(y0.min()-1, y0.max()+1, 500)
    new_xgrid, new_ygrid = np.meshgrid(xnew, ynew, indexing='ij')
    arr_ = linfit(beta, np.array([new_xgrid, new_ygrid]))
    ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                      coords=dict(x=xnew, y=ynew))


    arr.plot.imshow(ax=axs[0,id], cmap=cmap, **kwarg)

    ang_.vza.plot.imshow(ax=axs[1,id], cmap=cmap, **kwarg)
    axs[1,id].set_title('fitted 2D-function')

axs[0,0].set_xlim((0, 22))
axs[0,0].set_ylim((0, 22))

plt.savefig('./fig/example_2D_fitting_one_band.png', dpi=300)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True, sharey=True)
axs = axs.ravel()
fig.subplots_adjust(bottom=0.05, top=0.965, left=0.05, right=0.98,
                    hspace=0.12, wspace=0.1)

id=1
arr=dx.isel(bandId=0,detectorId=id).dropna('y', how='all').dropna('x', how='all')
x0,y0 = arr.x.values,arr.y.values
xgrid, ygrid = np.meshgrid(x0, y0, indexing='ij')
arr_ = linfit(beta, np.array([xgrid, ygrid]))
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                  coords=dict(x=x0, y=y0))

kwarg = dict(vmin=arr.min(), vmax=arr.max())
arr.plot.imshow(ax=axs[0], cmap=cmap, **kwarg)
axs[0].set_title('raw data')
ang_.vza.plot.imshow(ax=axs[1], cmap=cmap, **kwarg)
axs[1].set_title('fitted 2D-function (raw grid)')
residual = arr - ang_.vza
residual.plot.imshow(ax=axs[2], cmap=plt.cm.RdBu)
axs[2].set_title('Residuals')

xnew = np.linspace(x0.min(), x0.max(), 500)
ynew = np.linspace(y0.min(), y0.max(), 500)
new_xgrid, new_ygrid = np.meshgrid(xnew, ynew, indexing='ij')
arr_ = linfit(beta, np.array([new_xgrid, new_ygrid]))
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                  coords=dict(x=xnew, y=ynew))
ang_.vza.plot.imshow(ax=axs[3], cmap=cmap, **kwarg)
axs[3].set_title('2D-function in new grid')
plt.savefig('./fig/example_2D_fitting_one_detector.png', dpi=300)

dims = xgrid.size
shape = xgrid.shape
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
axs = axs.ravel()
for i, theta in enumerate([0, 70, 75, 76, 77, 76]):
    cos_, sin_ = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    xgrid_prime = xgrid * cos_ + ygrid * sin_
    ygrid_prime = -xgrid * sin_ + ygrid * cos_
    axs[i].scatter(xgrid_prime, ygrid_prime, c=arr.values, cmap=cmap)
    axs[i].set_title(str(theta))

# rotate x, y
theta = 76
xgrid_prime, ygrid_prime = rot_plane(xgrid, ygrid, theta)

# vectorize
values = arr.values.flatten()
x_prime = xgrid_prime.flatten()
y_prime = ygrid_prime.flatten()

# remove NaN
idx = ~np.isnan(values)
values = values[idx]
points = np.empty((2, len(values)))
points[0] = x_prime[idx]
points[1] = y_prime[idx]

# interp = RBFInterpolator(points.T,values,kernel='linear')
# new_points = np.vstack([x_prime,y_prime])
# arr_ =interp(new_points.T).reshape([23,7])
# ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
#                       coords=dict(x=x0, y=y0))

# set ODR fitting
mean = np.nanmean(values)
linear = odr.Model(linfit)
data = odr.Data(points, values)
beta0 = [0, 0, mean]
fit = odr.ODR(data, linear, beta0=beta0, taufac=0.01, ndigit=3)
fit.set_job(fit_type=0)
resfit = fit.run()
resfit.pprint()
beta = resfit.beta

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True, sharey=True)
axs = axs.ravel()
fig.subplots_adjust(bottom=0.05, top=0.965, left=0.05, right=0.98,
                    hspace=0.12, wspace=0.1)

arr_ = linfit(beta, np.array([xgrid_prime, ygrid_prime]))
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                  coords=dict(x=x0, y=y0))

kwarg = dict(vmin=arr.min(), vmax=arr.max())
arr.plot.imshow(ax=axs[0], cmap=cmap, **kwarg)
axs[0].set_title('raw data')
ang_.vza.plot.imshow(ax=axs[1], cmap=cmap, **kwarg)
axs[1].set_title('fitted 2D-function (raw grid)')
residual = arr - ang_.vza
residual.plot.imshow(ax=axs[2], cmap=plt.cm.RdBu)
axs[2].set_title('Residuals')

xnew = np.linspace(x0.min(), x0.max(), 500)
ynew = np.linspace(y0.min(), y0.max(), 500)

# rotate
new_xgrid, new_ygrid = np.meshgrid(xnew, ynew, indexing='ij')
new_xgrid_prime, new_ygrid_prime = rot_plane(new_xgrid, new_ygrid, theta)

arr_ = linfit(beta, np.array([new_xgrid_prime, new_ygrid_prime]))
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                  coords=dict(x=xnew, y=ynew))
ang_.vza.plot.imshow(ax=axs[3], cmap=cmap, **kwarg)
axs[3].set_title('2D-function in new grid')
plt.savefig('./fig/example_2D_fitting_one_detector.png', dpi=300)

# illustration roatation in 3D
from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.scatter3D(xgrid, ygrid, arr.values, s=14)
ax.set_title('raw grid')
ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.scatter3D(xgrid_prime, ygrid_prime, arr.values, s=14)
ax.set_title('projected grid')
ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.scatter3D(xgrid_prime, ygrid_prime, arr.values, s=14)
ax.scatter3D(new_xgrid, new_ygrid, ang_.vza.values, s=2, label='2D-function')
ax.set_title('2D-function fitting')

arr_ = linfit(beta, np.array([new_xgrid, new_ygrid]))

ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                  coords=dict(x=xnew, y=ynew))

xnew = np.linspace(x0.min(), x0.max(), 50)
ynew = np.linspace(y0.min(), y0.max(), 50)

# rotate
new_xgrid, new_ygrid = np.meshgrid(xnew, ynew, indexing='ij')
new_xgrid_prime, new_ygrid_prime = rot_plane(new_xgrid, new_ygrid, theta)

arr_ = linfit(beta, np.array([new_xgrid_prime, new_ygrid_prime]))
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                  coords=dict(x=xnew, y=ynew))

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharex=True, sharey=True)
axs = axs.ravel()
fig.subplots_adjust(bottom=0.1, top=0.965, left=0.1, right=0.98,
                    hspace=0.075, wspace=0.1)

kwarg = dict(vmin=arr.min(), vmax=arr.max())

arr.plot.imshow(ax=axs[3], cmap=cmap, **kwarg)
arr.plot.imshow(ax=axs[0], cmap=cmap, **kwarg)
arr_ = linfit(beta, np.array([new_xgrid, new_ygrid])).T
arr_[np.isnan(arr.values)] = np.nan

ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                  coords=dict(x=xnew, y=ynew))

ang_.vza.plot.imshow(ax=axs[1], cmap=cmap, **kwarg)
residual = arr - ang_.vza
residual.plot.imshow(ax=axs[2], cmap=cmap, )

# beta_=np.array([-0.004,-0.015])-0.005
arr_ = linfit(beta_, np.array([xgrid, ygrid])).T
arr_[np.isnan(arr.values)] = np.nan
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                  coords=dict(x=x0, y=y0))
ang_.vza.plot.imshow(ax=axs[4], cmap=cmap, **kwarg)

residual = arr - ang_.vza
residual.plot.imshow(ax=axs[5], origin='upper', cmap=cmap)
axs[-1].set_xlim((0, 22))

# ---------------------------------
# test with RegularGridInterpolator
# ---------------------------------

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.985, left=0.1, right=0.98,
                    hspace=0.01, wspace=0.025)
for id in range(5):
    arr = view_ang.vza.isel(bandId=0, detectorId=id).dropna('y', how='all').dropna('x', how='all')
    interp = RegularGridInterpolator((arr.x.values, arr.y.values), arr.values, bounds_error=False, fill_value=None)

    interp = RBFInterpolator((arr.x.values, arr.y.values), arr.values)

    values = arr.values.flatten()
    xgrid, ygrid = np.meshgrid(arr.x, arr.y)
    x_ = np.linspace(arr.x.min(), arr.x.max(), 500)
    y_ = np.linspace(arr.y.min(), arr.y.max(), 500)
    new_xgrid, new_ygrid = np.meshgrid(x_, y_)
    arr_ = interp((new_xgrid, new_ygrid)).T
    ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                      coords=dict(x=x_, y=y_))  # .interp(x=x_full,y=y_full,method='nearest')

    arr.plot(ax=axs[0])
    ang_.vza.plot(ax=axs[1])

dims = xgrid.size
points = np.empty((dims, 2))
points[:, 0] = xgrid.flatten()
points[:, 1] = ygrid.flatten()
grid = np.stack((arr.x, arr.y), axis=1)

linmod = scipy.odr.Model(linfit)
data = scipy.odr.Data(points, values)
odrfit = scipy.odr.ODR(data, linmod, beta0=[1., 1., 1.])
odrres = odrfit.run()
odrres.pprint()

interp = LinearNDInterpolator(points, values)
interpolated_values = griddata(points, values, (new_xgrid, new_ygrid), method='linear')

grid = arr.values
x, y = np.indices(grid.shape)
xvalid = x[~np.isnan(grid)]
xvalid = x[~np.isnan(grid)]
reg = LinearRegression().fit(
    np.stack((x[~np.isnan(grid)], y[~np.isnan(grid)]), axis=1),
    grid[~np.isnan(grid)])
grid_filled = reg.predict(np.stack((x.ravel(), y.ravel()),
                                   axis=1)).reshape(grid.shape)
out_grid[np.isnan(grid)] = grid_filled[np.isnan(grid)]

arr_ = arr.interp(x=new_x, y=new_y, method=method)
