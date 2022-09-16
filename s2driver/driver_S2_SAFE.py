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
import eoreader.bands as eobands

plt.ioff()
save = False

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen', 'orange', 'red'])
opj = os.path.join
imageSAFE = '/sat_data/satellite/sentinel2/L1C/31TFJ/S2A_MSIL1C_20201004T104031_N0209_R008_T31TFJ_20201004T125253.SAFE'
imageSAFE = '/sat_data/satellite/sentinel2/L1C/31TFJ/S2B_MSIL1C_20220731T103629_N0400_R008_T31TFJ_20220731T124834.SAFE'


# TODO put it as class

abspath = os.path.abspath(imageSAFE)
dirroot, basename = os.path.split(abspath)


# --------------------------------
# define interpolation parameters
# --------------------------------
# tile for 10m resolution: width,height = 10980,10980
# tile for 20m resolution: width,height = 5490,5490
# tile for 60m resolution: width,height = 1830,1830

resolution = 60
width, height = 1830, 1830  # 5490, 5490
resolution = 20
width, height = 5490, 5490

xml_granule = glob.glob(opj(imageSAFE, 'GRANULE', '*', 'MTD_TL.xml'))[0]
xml_file = glob.glob(opj(imageSAFE, 'MTD*.xml'))[0]

BAND_NAMES = np.array(['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A','B09','B10' ,'B11', 'B12'])
BAND_NAMES_EOREADER = np.array(['CA','BLUE','GREEN','RED','VRE_1',
                                'VRE_2','VRE_3','NARROW_NIR','NIR',
                                'WV','SWIR_CIRRUS','SWIR_1','SWIR_2'])

BAND_ID = [b.replace('B', '') for b in BAND_NAMES]
NATIVE_RESOLUTION = [60,10,10,10,20,20,20,10,20,60,60,20,20]

# select band to process and load
band_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
#band_idx = [ 1, 2, 3]

norm = mpl.colors.Normalize(vmin=0, vmax=11)

reader = Reader()
# Open the product
prod = reader.open(imageSAFE, remove_tmp=True)
processing_baseline = prod._processing_baseline
extent = prod.extent()
bounds = extent.bounds
minx, miny, maxx, maxy = bounds.values[0]
crs = prod.crs()

#----------------------------------
# getting bands
#----------------------------------

bands = prod.stack(list(BAND_NAMES_EOREADER[band_idx]),resolution=resolution)
bands = bands.rename({'z': 'bandID'})
bands = bands.assign_coords(bandID=list(BAND_NAMES[band_idx]))
prod.clear()

plt.figure()
fig = bands[:, ::10, ::10].plot(col='bandID',col_wrap=4,aspect=1.15,robust=True,cmap=plt.cm.binary_r)
for ax in fig.axes.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
plt.savefig('./fig/example_reflectance_all_bands.png',dpi=300)
#plt.show()

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

indexing = 'xy'

new_x = np.linspace(minx, maxx, width)
new_y = np.linspace(miny, maxy, height)[::-1]



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

detector_mask_name = 'DETFOO'
betas = np.full((detector_num,3),np.nan)
arr = np.zeros((Nx,Ny),dtype=np.uint16)
def get_band_angle_as_numpy(xarr, bandId=0, resolution=20, verbose=True):
    #xarr = view_ang.vza
    detector_offset = xarr.detectorId.values.min()
    mask = _open_mask(detector_mask_name, BAND_ID[bandId], resolution=resolution).astype(np.int8)
    mask = mask.squeeze()

    # TODO check how to avoid taking the nodata value "0" when coarsening the raster
    # TODO for the moment this induces bad detector number at the edge of the image swath
    # mask = _open_mask(detector_mask_name, BAND_ID[bandId], resolution=NATIVE_RESOLUTION[bandId])
    # # mask nodata value
    # mask = mask.where(mask!=0)
    # # resample mask at the desired resolution
    # if resolution != NATIVE_RESOLUTION[iband]:
    #     mask = mask.interp(x=new_x, y=new_y, method='nearest')
    # # compress mask into int8
    # mask = mask.astype(np.int8)


    x,y = mask.x.values,mask.y.values
    prod.clear()
    xarr_= xarr.sel(bandId=bandId)
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
    del mask
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
    ang = np.arccos(ang)
    return np.degrees(ang)

def get_all_band_angles(sun_ang,view_ang,band_idx,crs=None,resolution=20,method='linear'):

    # -----------------------------------------------------------------
    # Sun angles (easy!) based on standard bidimensional interpolation
    # -----------------------------------------------------------------
    new_sun_ang = sun_ang.interp(x=new_x, y=new_y, method=method)

    # ------------------------------------------------------
    # Viewing angles (not easy!) based on 2D-plane fitting
    # ------------------------------------------------------
    vza = view_ang.vza
    vazi = view_ang.vazi

    # ---------------------------------------------------------
    # convert vza, azi angles into cartesian vector components
    # (NOT NEEDED FOR THE MOMENT!!)
    # ---------------------------------------------------------
    # np.tan(np.deg2rad(view_ang.vza)) * np.sin(np.deg2rad(view_ang.vazi))
    # np.tan(np.deg2rad(view_ang.vza)) * np.cos(np.deg2rad(view_ang.vazi))
    # dx.name = 'dx'
    # dy.name = 'dy'

    new_vza, new_vazi = [], []
    for ibandId, bandId in enumerate(band_idx):
        print(bandId)
        new_vza.append(get_band_angle_as_numpy(vza, bandId=bandId,resolution=resolution))
        new_vazi.append(get_band_angle_as_numpy(vazi, bandId=bandId,resolution=resolution))

    new_ang = xr.Dataset(data_vars=dict(vza=(['bandID','y', 'x'], np.array(new_vza)),
                                         vazi=(['bandID','y', 'x'], np.array(new_vazi))),
                  coords=dict(band=BAND_NAMES[band_idx],x=new_x, y=new_y))
    new_ang['sza'] = new_sun_ang.sza
    new_ang['razi'] = new_ang.vazi - new_sun_ang.sazi
    #new_ang['scat_ang'] = scat_angle(new_ang.sza, new_ang.vza, new_ang.razi)

    if crs:
        set_crs(new_ang, crs)

    del new_vza, new_vazi, vza, vazi, new_sun_ang

    return new_ang

new_ang = get_all_band_angles(sun_ang,view_ang,band_idx,crs=crs,resolution=resolution)
del view_ang, sun_ang

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
#binary cmap
bcmap = ListedColormap(['khaki', 'lightblue'])
# Compute cartopy projection (EOReader always is in UTM)
extent = prod.extent()
str_epsg = str(extent.crs.to_epsg())
zone = str_epsg[-2:]
is_south = str_epsg[2] == 7
proj = ccrs.UTM(zone, is_south)

# Get extent values
# The extents must be defined in the form (min_x, max_x, min_y, max_y)
bounds = extent.bounds
extent_val = [bounds.minx[0], bounds.maxx[0], bounds.miny[0], bounds.maxy[0]]

# Compute NDWI
green = bands.sel(bandID='B03')
nir =bands.sel(bandID='B8A')
ndwi = (green - nir)/(green + nir)

def water_mask(ndwi,threshold=0):
    water = xr.where(ndwi > threshold, 1, 0)
    return water.where(~np.isnan(ndwi))

plt.figure(figsize=(20,6))
shrink=0.8
axes = plt.subplot(1, 3, 1, projection=proj)
axes.set_extent(extent_val, proj)
fig = ndwi[::10, ::10].plot.imshow( extent=extent_val, transform=proj, cmap=plt.cm.BrBG,robust=True,cbar_kwargs = { 'shrink':shrink})
#axes.coastlines(resolution='10m',linewidth=1)
axes.set_title('Sentinel 2, NDWI')

axes = plt.subplot(1, 3, 2, projection=proj)
water = water_mask(ndwi,  0.2)
axes.set_extent(extent_val, proj)
water[ ::10, ::10].plot.imshow( extent=extent_val, transform=proj, cmap=bcmap, cbar_kwargs={'ticks': [0,1],'shrink':shrink})
axes.set_title('0.2 < NDWI')

axes = plt.subplot(1, 3, 3, projection=proj)
water = water_mask(ndwi,  0.)
axes.set_extent(extent_val, proj)
water[ ::10, ::10].plot.imshow( extent=extent_val, transform=proj, cmap=bcmap, cbar_kwargs={'ticks': [0,1],'shrink':shrink})
axes.set_title('0. < NDWI')

plt.savefig('./fig/example_ndwi_mask.png',dpi=300)

plt.show()

plt.figure()
fig = new_ang.scat_ang.plot(col='band', cmap=cmap,col_wrap=4,aspect=1.15,robust=True)
for ax in fig.axes.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
plt.savefig('./fig/example_scattering_angle_all_bands.png',dpi=300)
#plt.show()


