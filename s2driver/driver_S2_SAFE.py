import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
from numba import jit

import pandas as pd
import geopandas as gpd
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

plt.ioff()
save = False

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen', 'orange', 'red'])
opj = os.path.join
imageSAFE = '/sat_data/satellite/sentinel2/L1C/31TFJ/S2A_MSIL1C_20201004T104031_N0209_R008_T31TFJ_20201004T125253.SAFE'
imageSAFE = '/sat_data/satellite/sentinel2/L1C/31TFJ/S2B_MSIL1C_20220731T103629_N0400_R008_T31TFJ_20220731T124834.SAFE'

abspath = os.path.abspath(imageSAFE)
dirroot, basename = os.path.split(abspath)

res = 20
xml_granule = glob.glob(opj(imageSAFE, 'GRANULE', '*', 'MTD_TL.xml'))[0]
xml_file = glob.glob(opj(imageSAFE, 'MTD*.xml'))[0]

BAND_NAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
BAND_ID = [b.replace('B', '') for b in BAND_NAMES]
band_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

norm = mpl.colors.Normalize(vmin=0, vmax=11)

reader = Reader()
# Open the product
prod = reader.open(imageSAFE, remove_tmp=True)
processing_baseline = prod._processing_baseline
extent = prod.extent()
bounds = extent.bounds
minx, miny, maxx, maxy = bounds.values[0]
crs = prod.crs()

# ---------------------------------
# getting detector masks
# ---------------------------------
# TODO generalize to load all available masks
if processing_baseline < 4:
    _open_mask = prod._open_mask_lt_4_0
else:
    _open_mask = prod._open_mask_gt_4_0

# mask_names = eo.products.optical.s2_product.S2GmlMasks.list_values()
# mask_names = ['DETFOO']
# masks_ = []
# for mask_name in mask_names:
#     for i, id in enumerate(BAND_ID):
#         print(id, mask_name)
#         try:
#             mask_ = _open_mask(mask_name, id, res=res).astype(np.int8)
#             mask_.assign_coords(band=[id])
#             prod.clear()
#         except:
#             break
#         if len(mask_) == 0:
#             continue
#         masks_.append(mask_)
#
#         name = mask_.gml_id.str.replace('detector_footprint-', '')
#         name = name.str.split('-', expand=True).values
#         detectorId = (name[:, 2]).astype(int)
#         mask_['bandId'], mask_['detectorId'] = name[:, 0], detectorId
#         mask_ = mask_.set_index(['bandId', 'detectorId'])
#         masks_.append(mask_['geometry'])
#
# detector_num = np.max(detectorId) + 1
# masks = pd.concat(masks_)
# xrmasks = masks.to_xarray()
# plt.plot(*masks.geometry.isel(band=0).values[4].exterior.xy)

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

# compute x and y for angle grids
# check dimension
Nx, Ny = sza.shape

xang = np.linspace(minx, maxx, Nx)
yang = np.linspace(miny, maxy, Ny)[::-1]


def set_crs(arr, crs):
    arr.rio.set_crs(crs, inplace=True)
    arr.rio.write_crs(inplace=True)


sun_ang = xr.Dataset(data_vars=dict(sza=(['y', 'x'], sza),
                                    sazi=(['y', 'x'], sazi)),
                     coords=dict(x=xang, y=yang))
set_crs(sun_ang, crs)

# ---------------------------------
# getting viewing geometry datacube
# ---------------------------------
bandIds, detectorIds = [], []
for angleID in root.findall('.//Tile_Angles/Viewing_Incidence_Angles_Grids'):
    bandIds.append(int(angleID.attrib['bandId']))
    detectorIds.append(int(angleID.attrib['detectorId']))
Nband, Ndetector = np.max(bandIds) + 1, np.max(detectorIds) + 1

# allocate/fill rasters
vza, vazi = np.full((Nband, Ndetector, Nx, Ny), np.nan, dtype=float), np.full((Nband, Ndetector, Nx, Ny), np.nan,
                                                                              dtype=float)

for angleID in root.findall('.//Tile_Angles/Viewing_Incidence_Angles_Grids'):
    iband = int(angleID.attrib['bandId'])
    idetector = int(angleID.attrib['detectorId'])
    vza[iband, idetector] = parse_angular_grid_node(angleID.find('Zenith'))
    vazi[iband, idetector] = parse_angular_grid_node(angleID.find('Azimuth'))

view_ang = xr.Dataset(data_vars=dict(vza=(['bandId', 'detectorId', 'y', 'x'], vza),
                                     vazi=(['bandId', 'detectorId', 'y', 'x'], vazi)),
                      coords=dict(bandId=range(Nband),
                                  detectorId=range(Ndetector),
                                  x=xang, y=yang))
set_crs(view_ang, crs)

# clean up Dataset (remove empty slices)
param = 'vza'
view_ang = view_ang.dropna('detectorId', how='all')

# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharex=True, sharey=True)
# fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.98,
#                     hspace=0.15, wspace=0.1)
# axs = axs.ravel()
# vmin, vmax = view_ang[param].min(), view_ang[param].max()
# for id in range(5):
#     view_ang[param].isel(bandId=0, detectorId=id).plot(ax=axs[id], cmap=cmap, vmin=vmin, vmax=vmax)
#
#     axs[id].plot(*masks[BAND_NAMES[0], id].exterior.xy, color='black')
#     axs[id].set(xticks=[], yticks=[])
# axs[-1].set_visible(False)
# if save:
#     plt.savefig('./fig/example_detector_raw_angle_' + param + '.png', dpi=300)

# -------------------------
# interpolation
# -------------------------

# --------------------------------
# define interpolation parameters
# --------------------------------
# tile for 10m resolution: width,height = 10980,10980
# tile for 20m resolution: width,height = 5490,5490
# tile for 60m resolution: width,height = 1830,1830
indexing = 'xy'
resolution = 60
width, height = 1830, 1830  # 5490, 5490
new_x = np.linspace(minx, maxx, width)
new_y = np.linspace(miny, maxy, height)
new_xgrid, new_ygrid = np.meshgrid(new_x, new_y, indexing=indexing)

# define the new grids in xarray for further clipping facilities
coords_arr = xr.Dataset(data_vars=dict(xgrid=(['y', 'x'], new_xgrid),
                                       ygrid=(['y', 'x'], new_ygrid)),
                        coords=dict(x=new_x, y=new_y))
set_crs(coords_arr, crs)

# -----------------------------------------------------------------
# Sun angles (easy!) based on standard bidimensional interpolation
# -----------------------------------------------------------------
method = 'linear'
new_sun_ang = sun_ang.interp(x=new_x, y=new_y, method=method)

# ------------------------------------------------------
# Viewing angles (not easy!) based on 2D-plane fitting
# ------------------------------------------------------
# convert vza, azi angles into cartesian vector components
dx = np.tan(np.deg2rad(view_ang.vza)) * np.sin(np.deg2rad(view_ang.vazi))
dy = np.tan(np.deg2rad(view_ang.vza)) * np.cos(np.deg2rad(view_ang.vazi))
dx.name = 'dx'
dy.name = 'dy'


# dx.isel(bandId=0).plot.imshow(col='detectorId', col_wrap=3, origin='upper', cmap=cmap)
# dy.isel(bandId=0).plot.imshow(col='detectorId', col_wrap=3, origin='upper', cmap=cmap)

# ---------------------------------
# test with ODR multilinear regression
# ---------------------------------

def linfit(beta, x):
    return beta[0] * x[0] + beta[1] * x[1] + beta[2]

@jit(nopython=True)#"uint16[:,:](float64[:],float64[:],float64[:,:],float64[:,:],intp,intp)",
def lin2D(arr,x,y,mask,betas,detector_offset=0,scale_factor=100):

    Nx,Ny = mask.shape

    for ii in range(Nx):
        for jj in range(Ny):
            detect = mask[ii,jj]
            if detect == 0:
                continue
            beta = betas[detect - detector_offset]
            val = beta[0] * x[jj] + beta[1] * y[ii] + beta[2]
            # compression using simple int8 and scale factor
            arr[ii,jj] = (val * scale_factor)


def data_fitting(x0, y0, arr, verbose=False, indexing='xy'):
    xgrid, ygrid = np.meshgrid(x0, y0, indexing=indexing)

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


detector_num = 5
bandId = 2
detector_mask_name = 'DETFOO'
betas = np.full((detector_num,3),np.nan)
arr = np.zeros((Nx,Ny),dtype=np.uint16)
def get_band_angle_as_numpy(xarr, bandId=0, resolution=20, verbose=True):
    #xarr = view_ang.vza
    detector_offset = xarr.detectorId.values.min()
    mask = _open_mask(detector_mask_name, BAND_ID[bandId], resolution=resolution).astype(np.int8)
    mask = mask.squeeze()
    x,y = mask.x.values,mask.y.values
    prod.clear()
    xarr_= xarr.isel(bandId=bandId)
    for id in range(detector_num):
        # --------------------------------------------------------------
        # Linear 2D-fitting to get the function of the regression plane
        # --------------------------------------------------------------
        arr = xarr_.isel(detectorId=id).dropna('y', how='all').dropna('x', how='all')
        x0, y0 = arr.x.values, arr.y.values
        betas[id,:] = data_fitting(x0, y0, arr, verbose=verbose)

        # [bandId,id,'dx',*beta]
        # --------------------------------------------------------------
        # Get detector coordinates from masking with detectorId info
        # --------------------------------------------------------------
    # compression in uint16 (NB: range 0-65535)
    new_arr = np.full((width, height), np.nan,dtype=np.float32)
    lin2D(new_arr,x,y,mask.__array__(),betas,detector_offset=detector_offset,scale_factor=1)
    # plt.figure()
    # plt.imshow(new_arr, cmap=cmap)
    # plt.colorbar()
    # plt.show()

    return new_arr

def scat_angle(sza, vza, azi):
    '''
    self.azi: azimuth in rad for convention azi=180 when sun-sensor in opposition
    :return: scattering angle in deg
    '''

    sza = np.radians(sza)
    vza = np.radians(vza)
    azi = np.radians(azi)
    ang = -np.cos(sza) * np.cos(vza) - np.sin(sza) * np.sin(vza) * np.cos(azi)
    ang = np.arccos(ang) * 180 / np.pi
    return ang

def get_all_band_angles(view_ang,resolution=20):
    # ---------------------------------------------------------
    # convert vza, azi angles into cartesian vector components
    # ---------------------------------------------------------
    dx = view_ang.vza #np.tan(np.deg2rad(view_ang.vza)) * np.sin(np.deg2rad(view_ang.vazi))
    dy = view_ang.vazi #np.tan(np.deg2rad(view_ang.vza)) * np.cos(np.deg2rad(view_ang.vazi))
    dx.name = 'dx'
    dy.name = 'dy'
    new_dx, new_dy = [], []
    for ibandId, bandId in enumerate(band_idx):
        print(bandId)
        new_dx.append(get_band_angle_as_numpy(dx, bandId=ibandId,resolution=resolution))
        new_dy.append(get_band_angle_as_numpy(dy, bandId=ibandId,resolution=resolution))

    vza,vazi = np.array(new_dx), np.array(new_dy)
    # vazi = np.arctan2(new_dy,new_dx)
    # vza = np.degrees(np.arctan2( new_dy/10000,np.cos(vazi)))

new_view_ang = xr.Dataset(data_vars=dict(vza=(['band','y', 'x'], vza),
                                         vazi=(['band','y', 'x'], vazi)),
                  coords=dict(band=BAND_NAMES,x=new_x, y=new_y[::-1]))
set_crs(new_view_ang, crs)
plt.figure()
new_view_ang.vazi.plot(col='band', cmap=cmap,col_wrap=4,robust=True)
plt.show()

detarr = ang_.vza.rio.clip([mask_])
arr_new.append(detarr)

arr_new = xr.merge(arr_new)
arr_new.vza.plot()

# ---------------------------------
# 3D plotting example
fig = plt.figure(figsize=(20, 10))
arr = xarr.isel(bandId=0, detectorId=id).dropna('y', how='all').dropna('x', how='all')
elev, azim = 70, 20
x0, y0 = arr.x.values, arr.y.values

ax = fig.add_subplot(1, 3, 1, projection='3d')
xgrid, ygrid = np.meshgrid(x0, y0, indexing='ij')
ax.scatter3D(xgrid, ygrid, arr.values, s=14)
ax.set_title('raw grid')
ax.view_init(elev=elev, azim=azim)

ax = fig.add_subplot(1, 3, 2, projection='3d')
xgrid, ygrid = np.meshgrid(x0, y0, indexing='xy')
ax.scatter3D(xgrid, ygrid, arr.values, s=14)
ax.set_title('projected grid')
ax.view_init(elev=elev, azim=azim)

ax = fig.add_subplot(1, 3, 3, projection='3d')
beta = data_fitting(x0, y0, arr, verbose=True)
xnew = np.linspace(x0.min() - 1, x0.max() + 1, 50)
ynew = np.linspace(y0.min() - 1, y0.max() + 1, 50)
new_xgrid, new_ygrid = np.meshgrid(xnew, ynew, indexing='xy')
arr_ = linfit(beta, np.array([new_xgrid, new_ygrid]))
ang_ = xr.Dataset(data_vars=dict(vza=(['y', 'x'], arr_)),
                  coords=dict(x=xnew, y=ynew))
ax.scatter3D(xgrid, ygrid, arr.values, s=18)
ax.scatter3D(new_xgrid, new_ygrid, ang_.vza.values, alpha=0.3, color='red', s=1.2, label='2D-function')
ax.set_title('2D-function fitting')
ax.view_init(elev=elev, azim=azim)
# ax.set_zlim(0.078, 0.092)
if save:
    plt.savefig('./fig/example_3D_fitting_one_detector_v2.png', dpi=300)

# -----------------------------------


kwarg = dict(vmin=xarr.min(), vmax=xarr.max())
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(22, 12), sharex=True, sharey=True)
fig.subplots_adjust(bottom=0.05, top=0.965, left=0.05, right=0.98,
                    hspace=0.12, wspace=0.1)
new_arr = np.full
for id in range(5):
    arr = xarr.isel(bandId=bandId, detectorId=id).dropna('y', how='all').dropna('x', how='all')
    x0, y0 = arr.x.values, arr.y.values

    beta = data_fitting(x0, y0, arr, verbose=True)

    xnew = np.linspace(x0.min() * (1 - dilation), x0.max() * (1 + dilation), 500)
    ynew = np.linspace(y0.min() * (1 - dilation), y0.max() * (1 + dilation), 500)

    new_xgrid, new_ygrid = np.meshgrid(xnew, ynew, indexing='xy')
    arr_ = linfit(beta, np.array([new_xgrid, new_ygrid]))
    ang_ = xr.Dataset(data_vars=dict(vza=(['y', 'x'], arr_)),
                      coords=dict(x=xnew, y=ynew))
    set_crs(ang_, crs)
    arr.plot.imshow(ax=axs[0, id], cmap=cmap, **kwarg)
    axs[1, id].plot(*masks[BAND_NAMES[bandId], id].exterior.xy, color='black')
    mask_ = masks[BAND_NAMES[bandId], id]
    ang_.vza.plot.imshow(ax=axs[1, id], cmap=cmap, **kwarg)
    axs[1, id].set_title('fitted 2D-function')
    detarr = ang_.vza.rio.clip([mask_])
    arr_new.append(detarr)
    detarr.plot(ax=axs[2, id], cmap=cmap, **kwarg)

axs[0, 0].set_xlim((minx, maxx))
axs[0, 0].set_ylim((miny, maxy))
plt.show()

plt.savefig('./fig/example_2D_fitting_one_band_v3.png', dpi=300)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True, sharey=True)
axs = axs.ravel()
fig.subplots_adjust(bottom=0.05, top=0.965, left=0.05, right=0.98,
                    hspace=0.12, wspace=0.1)

id = 1
arr = dx.isel(bandId=0, detectorId=id).dropna('y', how='all').dropna('x', how='all')
x0, y0 = arr.x.values, arr.y.values
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
