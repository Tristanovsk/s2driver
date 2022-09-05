

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
#import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')
from osgeo import gdal
#import rasterio as rio

opj = os.path.join
imageSAFE = '/sat_data/satellite/sentinel2/L1C/31TFJ/S2A_MSIL1C_20201004T104031_N0209_R008_T31TFJ_20201004T125253.SAFE/'
xml_granule = glob.glob(opj(imageSAFE, 'GRANULE', '*', 'MTD_TL.xml'))[0]
xml_file = glob.glob(opj(imageSAFE,  'MTD*.xml'))[0]

band_names = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12')
band_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

ds=gdal.Open(xml_file)
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

#-------------------------
# interpolation
#-------------------------
# tile for 10m resolution: width,height = 10980,10980
# tile for 20m resolution: width,height = 5490,5490
# tile for 60m resolution: width,height = 1830,1830

width,height = 5490,5490
new_x = np.linspace(0,Nx,width)
new_y = np.linspace(0,Ny,height)

# Sun angles (easy!)
method='linear'
new_sun_ang = sun_ang.interp(x=new_x,y=new_y,method=method)
new_sun_ang =new_sun_ang.assign_coords(x=range(width),y=range(height))
# fig,axs=plt.subplots(1,2,figsize=(10,5))
# new_sun_ang.sza.plot(ax=axs[0],cmap=plt.cm.Spectral_r)
# new_sun_ang.sazi.plot(ax=axs[1],cmap=plt.cm.Spectral_r)

# Viewing angles (not easy!)

for band, angs in view_ang.groupby('bandId'):
    for detec, angs_ in angs.groupby('detectorId'):
        print(band,detec)
        dx = np.tan(np.deg2rad(angs_.vza)) * np.sin(np.deg2rad(angs_.vazi))
        dy = np.tan(np.deg2rad(angs_.vza)) * np.cos(np.deg2rad(angs_.vazi))

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen','orange','red'])
view_ang.vza.isel(bandId=0).plot.imshow(col='detectorId', col_wrap=3,origin='upper',cmap=cmap)
view_ang.vazi.isel(bandId=0).plot.imshow(col='detectorId', col_wrap=3,origin='upper',cmap=cmap)

dx = np.tan(np.deg2rad(view_ang.vza)) * np.sin(np.deg2rad(view_ang.vazi))
dy = np.tan(np.deg2rad(view_ang.vza)) * np.cos(np.deg2rad(view_ang.vazi))
dx.isel(bandId=0).plot.imshow(col='detectorId', col_wrap=3,origin='upper',cmap=cmap)
dy.isel(bandId=0).plot.imshow(col='detectorId', col_wrap=3,origin='upper',cmap=cmap)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6),sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.985, left=0.1, right=0.98,
                    hspace=0.01, wspace=0.025)

for id in  range(5):
    dx.isel(bandId=0,detectorId=id).plot(ax=axs[0],cmap=cmap,vmin=dx.min(),vmax=dx.max())
    dy.isel(bandId=0,detectorId=id).plot(ax=axs[1],cmap=cmap,vmin=dy.min(),vmax=dy.max())


raw_res=5000
res=200
import scipy.odr as odr
from scipy.interpolate import griddata,LinearNDInterpolator,RBFInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

# ---------------------------------
# test with ODR multilinear regression
# ---------------------------------
def linfit(beta, x):

    return beta[0] * x[0] + beta[1] * x[1] + beta[2]

def rot_plane(x,y,theta):
    '''

    :param x: x-coordinates
    :param y: y_coordinates
    :param theta: rotation angle in degrees
    :return: x_prime-coordinates, y_prime-coordinates
    '''

    cos_,sin_ = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    x_prime = x * cos_ + y * sin_
    y_prime = -x * sin_ + y * cos_
    return x_prime, y_prime


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

id=1
arr=dx.isel(bandId=0,detectorId=id).dropna('y', how='all').dropna('x', how='all')
x0,y0 = arr.x.values,arr.y.values
xgrid, ygrid = np.meshgrid(x0, y0, indexing='ij')
beta = data_fitting(x0, y0, arr)

xnew = np.linspace(x0.min(), x0.max(), 500)
ynew = np.linspace(y0.min(), y0.max(), 500)
new_xgrid, new_ygrid = np.meshgrid(xnew, ynew, indexing='ij')
arr_ = linfit(beta, np.array([new_xgrid, new_ygrid]))
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                  coords=dict(x=xnew, y=ynew))


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True, sharey=True)
axs = axs.ravel()
fig.subplots_adjust(bottom=0.05, top=0.965, left=0.05, right=0.98,
                    hspace=0.12, wspace=0.1)

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


xgrid,ygrid=np.meshgrid(x0,y0,indexing='ij')
dims=xgrid.size
shape = xgrid.shape
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
axs=axs.ravel()
for i,theta in enumerate([0,70,75,76,77,76]):
    cos_,sin_ = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    xgrid_prime = xgrid * cos_ + ygrid * sin_
    ygrid_prime = -xgrid * sin_ + ygrid * cos_
    axs[i].scatter(xgrid_prime,ygrid_prime,c=arr.values,cmap=cmap)
    axs[i].set_title(str(theta))



# rotate x, y
theta=76
xgrid_prime,ygrid_prime = rot_plane(xgrid,ygrid,theta)


# vectorize
values = arr.values.flatten()
x_prime = xgrid_prime.flatten()
y_prime = ygrid_prime.flatten()

# remove NaN
idx=~np.isnan(values)
values = values[idx]
points = np.empty((2,len(values)))
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
data = odr.Data(points,values)
beta0 = [0,0,mean]
fit = odr.ODR(data, linear, beta0=beta0,taufac=0.01,ndigit=3)
fit.set_job(fit_type=0)
resfit = fit.run()
resfit.pprint()
beta =resfit.beta



fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),sharex=True,sharey=True)
axs=axs.ravel()
fig.subplots_adjust(bottom=0.05, top=0.965, left=0.5, right=0.98,
                    hspace=0.1, wspace=0.1)

arr_ = linfit(beta,np.array([xgrid_prime,ygrid_prime]))
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                      coords=dict(x=x0, y=y0))

kwarg=dict(vmin=arr.min(),vmax=arr.max())
arr.plot.imshow(ax=axs[0],cmap=cmap,**kwarg)

ang_.vza.plot.imshow(ax=axs[1],cmap=cmap,**kwarg)
residual = arr-ang_.vza
residual.plot.imshow(ax=axs[2],cmap=plt.cm.RdBu)

xnew = np.linspace(x0.min(),x0.max(),500)
ynew = np.linspace(y0.min(),y0.max(),500)

# rotate
new_xgrid, new_ygrid = np.meshgrid(xnew,ynew,indexing='ij')
new_xgrid_prime, new_ygrid_prime = rot_plane(new_xgrid, new_ygrid,theta)

arr_ = linfit(beta,np.array([new_xgrid_prime,new_ygrid_prime]))
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                      coords=dict(x=xnew, y=ynew))
ang_.vza.plot.imshow(ax=axs[3],cmap=cmap,**kwarg)


#beta = [1.78e-01, 1.78e-01, 6]
#
# beta0_ = [1,1]#,0.5*mean]
# def linfit_(x,b0,b1):
#     return b0 * x[0] + b1 * x[1] #+ b2
# popt, pcov = curve_fit(linfit_,points,values,p0=beta0_)

# recompute angle grid


arr_ = linfit(beta,np.array([new_xgrid, new_ygrid]))
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                      coords=dict(x=xnew, y=ynew))


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xgrid_prime,ygrid_prime,arr.values,s=14)
ax.scatter3D(new_xgrid,new_ygrid,ang_.vza.values,s=2)


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6),sharex=True,sharey=True)
axs=axs.ravel()
fig.subplots_adjust(bottom=0.1, top=0.965, left=0.1, right=0.98,
                    hspace=0.075, wspace=0.1)

kwarg=dict(vmin=arr.min(),vmax=arr.max())

arr.plot.imshow(ax=axs[3],cmap=cmap,**kwarg)
arr.plot.imshow(ax=axs[0],cmap=cmap,**kwarg)
arr_ = linfit(beta,np.array([new_xgrid, new_ygrid])).T
arr_[np.isnan(arr.values)]=np.nan

ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                      coords=dict(x=xnew, y=ynew))

ang_.vza.plot.imshow(ax=axs[1],cmap=cmap,**kwarg)
residual = arr-ang_.vza
residual.plot.imshow(ax=axs[2],cmap=cmap,)

#beta_=np.array([-0.004,-0.015])-0.005
arr_ = linfit(beta_,np.array([xgrid, ygrid])).T
arr_[np.isnan(arr.values)]=np.nan
ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                      coords=dict(x=x0, y=y0))
ang_.vza.plot.imshow(ax=axs[4],cmap=cmap,**kwarg)

residual = arr-ang_.vza
residual.plot.imshow(ax=axs[5],origin='upper',cmap=cmap)
axs[-1].set_xlim((0,22))

# ---------------------------------
# test with RegularGridInterpolator
# ---------------------------------

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6),sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.985, left=0.1, right=0.98,
                    hspace=0.01, wspace=0.025)
for id in  range(5):
    arr=view_ang.vza.isel(bandId=0,detectorId=id).dropna('y', how='all').dropna('x', how='all')
    interp = RegularGridInterpolator((arr.x.values,arr.y.values),arr.values,bounds_error=False, fill_value=None)

    interp = RBFInterpolator((arr.x.values,arr.y.values),arr.values)


    values = arr.values.flatten()
    xgrid,ygrid=np.meshgrid(arr.x,arr.y)
    x_ = np.linspace(arr.x.min(),arr.x.max(),500)
    y_ = np.linspace(arr.y.min(),arr.y.max(),500)
    new_xgrid, new_ygrid = np.meshgrid(x_,y_)
    arr_ = interp((new_xgrid, new_ygrid)).T
    ang_ = xr.Dataset(data_vars=dict(vza=(['x', 'y'], arr_)),
                      coords=dict(x=x_, y=y_))#.interp(x=x_full,y=y_full,method='nearest')


    arr.plot(ax=axs[0])
    ang_.vza.plot(ax=axs[1])

dims=xgrid.size
points = np.empty((dims, 2))
points[:, 0] = xgrid.flatten()
points[:, 1] = ygrid.flatten()
grid= np.stack((arr.x, arr.y), axis=1)

linmod = scipy.odr.Model(linfit)
data = scipy.odr.Data(points, values)
odrfit = scipy.odr.ODR(data, linmod, beta0=[1., 1., 1.])
odrres = odrfit.run()
odrres.pprint()

interp = LinearNDInterpolator(points,values)
interpolated_values = griddata(points, values, (new_xgrid, new_ygrid), method='linear')

grid =arr.values
x, y = np.indices(grid.shape)
xvalid = x[~np.isnan(grid)]
xvalid = x[~np.isnan(grid)]
reg = LinearRegression().fit(
    np.stack((x[~np.isnan(grid)], y[~np.isnan(grid)]), axis=1),
    grid[~np.isnan(grid)])
grid_filled = reg.predict(np.stack((x.ravel(), y.ravel()),
                                   axis=1)).reshape(grid.shape)
out_grid[np.isnan(grid)] = grid_filled[np.isnan(grid)]


arr_ = arr.interp(x=new_x,y=new_y,method=method)
