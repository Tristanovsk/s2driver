import os
from osgeo import gdal,ogr
import numpy as np
from xml.etree.ElementTree import ElementTree
from scipy import interpolate


def read_grid_as_array(grid_node):
    """
    Function that reads a grid of values from an xml file node
    into a NumPy array
    (c) Robin CREMESE, Magellium, 2021
    """

    grid_dim = len(grid_node)
    array = np.zeros((grid_dim, grid_dim))
    
    for row, values in enumerate(grid_node):
        array[row,:] = np.asarray(values.text.split(' ')).astype(float)
    return array


def get_detectors_footprint_mask(detfoo_path, output_raster_path, roi_x, roi_y, image_geotransform, row_min, row_max, col_min, col_max, mask):
    '''
    Read footprint mask from the given jp2 (raster) or gml (vector) file and crop it to the desired ROI.

    Arguments:
        detfoo_path         : (string) path/to/mask/filename.jp2 (.gml) - path to the footprint mask of the desired detector
        output_raster_path  : (string) path/to/raster/detfoo_mask.tif - raster that will be created if the mask is initially in the vector format
        roi_x               : UTM X coordinates of the ROI (4 corners)
        roi_y               : UTM Y coordinates of the ROI (4 corners)
        image_geotransform  : gdal geotransform variable of the image of the same spectral band that detector footprint mask
        row_min             : row number in image coordinates (first row starting at 0, and from the up of the image) of the ROI's upper border
        row_max             : row number in image coordinates (first row starting at 0, and from the up of the image) of the ROI's lower border
        col_min             : column number in image coordinates (first column starting at 0 and on the left) of the ROI's leftmost border
        col_max             : column number in image coordinates (first column starting at 0 and on the left) of the ROI's rightmost border
        mask                : numpy.array with ones and NaNs of the ROI shape, where 1 means that the pixel is in the ROI,
                                and NaN means that the pixel is outside of the ROI

    Output:
        detectors_matrix    : numpy.array of the mask shape, where for each pixel the value corresponds to the detector ID
                                which covers this pixel, and the valus is NaN is in the mask the value is NaN

    '''
    # Check the extension of the file
    _, fileext = os.path.splitext(detfoo_path)

    # If it is a gml vector file, the idea is to first rasterize the mask and give detector
    # ID values to thei corresponding pixels, and then to extract pixels of the ROI from this raster
    if fileext=='.gml':
        indriver    = ogr.GetDriverByName('GML')
        vec_ds      = indriver.Open(detfoo_path)
        # Create an output datasource in memory
        outdriver   = ogr.GetDriverByName('MEMORY')
        out_ds      = outdriver.CreateDataSource('memData')
        tmp         = outdriver.Open('memData',1)
        # Copy a layer to memory
        out_layer   = out_ds.CopyLayer(vec_ds.GetLayerByIndex(0),'det_foo',['OVERWRITE=YES'])
        # Create new field (atribute)
        fldDef      = ogr.FieldDefn('detNumber', ogr.OFTInteger)
        out         = out_layer.CreateField(fldDef)
        for feat in out_layer:
            id_str = feat.GetField(0)
            id_int = int(id_str.split('-')[2])
            feat.SetField(2,id_int)
            out_layer.SetFeature(feat)
        # Create ROI polygon
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x,y in zip(roi_x,roi_y):
            ring.AddPoint(x,y)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        # Create layer for the polygon
        poly_layer  = out_ds.CreateLayer(str('roi_polygon'),out_layer.GetSpatialRef(),ogr.wkbPolygon)
        idField     = ogr.FieldDefn("id", ogr.OFTInteger)
        poly_layer.CreateField(idField)
        featureDefn = poly_layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(poly)
        feature.SetField("id", 1)
        poly_layer.CreateFeature(feature)
        x_min, x_max, y_min, y_max = poly_layer.GetExtent()
        # Clip vectors
        out_clip_layer  = out_ds.CreateLayer(str('clipped_foo'),out_layer.GetSpatialRef())
        ogr.Layer.Clip(out_layer, poly_layer, out_clip_layer, ['OVERWRITE=YES'])
        # Rasterize vector layer
        ncol, nrow      = mask.shape[1], mask.shape[0]
        drv_tiff        = gdal.GetDriverByName("GTiff")
        out_raster_ds   = drv_tiff.Create(output_raster_path, ncol, nrow, 1, gdal.GDT_Byte)
        out_raster_ds.SetGeoTransform((x_min, image_geotransform[1], 0, y_max, 0, image_geotransform[5]))
        out_raster = out_raster_ds.GetRasterBand(1)
        # out_raster.Fill(0)
        status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                     [1],  # output to our new dataset's first band
                                     out_clip_layer,  # rasterize this layer
                                     None, None,  # don't worry about transformations since we're in same projection
                                     [0],  # burn value 0
                                     ['ALL_TOUCHED=TRUE',  # rasterize all pixels touched by polygons
                                      'ATTRIBUTE=detNumber']  # put raster values according to the 'id' field values
                                     )
        detectors_matrix = out_raster.ReadAsArray()
        # plt.figure(), plt.imshow(detectors_matrix, cmap='Paired'), plt.colorbar(), plt.title(os.path.basename(detfoo_path))
        out_raster_ds = None
        return detectors_matrix
    # If the file is the jp2 raster, then we just read it and extract pixels of the ROI
    if fileext=='.jp2':
        detectors_matrix = gdal.Open(detfoo_path).ReadAsArray()
        detectors_matrix = detectors_matrix[row_min:row_max+1,col_min:col_max+1]
        return detectors_matrix


def get_sun_angle_information(angles_xml, coin_X, coin_Y, mask, geotransform, band_id):
    '''
    Interpolate Sentinel-2 sun zenith and azimuth angles to the band_id native spatial resolution.

    Arguments:
        angles_xml      : (string) path to the MTD_TL.xml file which contains values of the sun angles.
        coin_X          : (number) UTM X coordinates of the upper left corner of te desired ROI to extract from the entire tile.
        coin_Y          : (number) UTM Y coordinates of the upper left corner of te desired ROI to extract from the entire tile.
        mask            : (numpy.array) array of the same shape as the ROI;
                            value 1 means that the coresponding pixel is in the ROI;
                            value numpy.nan means that the pixels is outside of the ROI;
                            this array is used to mask values outside of the ROI after the interpolation.
                            If you don't want to use this "feature" you can pass -1 for this argument.
        geotransform    : gdal GeoTransform variable (from gdal GetGeoTransform() function) which holds the geotransform of the jp2 S2 image with desired resolution
        band_id         : (int) index of the desired S2 band, starting at 0 (Band 1 -> band_id=0, Band 2 -> band_id=1 etc)

    Output:
        sza, saa : numpy.arrays with the interpolated values for the given ROI
    '''

    # Get the UTM coordinates of the upper left corner of the Sun Angle matrix (not the S2 image tile)
    root            = ElementTree().parse(angles_xml)
    node            = root.find('.//Geoposition[@resolution="%d"]' %geotransform[1])
    ulx_tile        = float(node[0].text)
    uly_tile        = float(node[1].text)

    # Final desired spatial resolution
    res_im          = int(geotransform[1])

    # Spatial resolution of the Sun angle data
    res_x_angles = 5000
    if geotransform[5]>0: # if in image geotransform the resolution along Y axis is > 0, then the angle resolution will be also > 0 (south-up images)
        res_y_angles = 5000
    else: # north-up images
        res_y_angles = -5000

    # Nodes of the XML for the Sun Zenith and Azimuth angles
    SunZenithNode = root.find('.//Sun_Angles_Grid/Zenith')
    SunAzimuthNode = root.find('.//Sun_Angles_Grid/Azimuth')

    # ==========================================================================
    # Sun Zenith angles
    # ==========================================================================
    
    # Read sun zenith angle values from xml into numpy.array
    zenith_array        = read_grid_as_array(SunZenithNode[2])

    # Mask NaNs in the array in case when the tile does not take the entire image space and these NaNs come from the original XML data
    zenith_array        = np.ma.array(zenith_array, mask=np.isnan(zenith_array))

    # Final image resolution
    res_x_im, res_y_im  = geotransform[1], geotransform[5]

    # Compute UTM coordinates for each pixel of the ROI
    # Eventually if you want to compute angles for every pixel in the image and you don't really have an ROI,
    # your coin_X and coin_Y can be UTM coordinates of the upper left corner of your tile and mask can be an array of the same shape as the tile
    # filled entirely with ones.
    x_im, y_im          = np.meshgrid(np.arange(coin_X+res_x_im/2, coin_X + mask.shape[1]*res_x_im,res_x_im),
                                      np.arange(coin_Y+res_y_im/2, coin_Y + mask.shape[0]*res_y_im,res_y_im))

    # Comput UTM coordinates for every Sun angle value
    x_utm,y_utm         = np.meshgrid(np.arange(ulx_tile, ulx_tile + zenith_array.shape[1]*res_x_angles,res_x_angles),
                                      np.arange(uly_tile, uly_tile + zenith_array.shape[0]*res_y_angles,res_y_angles))

    # Mask angle coordinates where the angle value is NaN
    x_utm               = np.ma.array(x_utm, mask=zenith_array.mask)
    y_utm               = np.ma.array(y_utm, mask=zenith_array.mask)

    # Interpolate sun angle values
    couche_interp       = interpolate.griddata((x_utm[~x_utm.mask].ravel(),                 # valid (not masked) UTM X coordinates of the sun angle values in vector format (.ravel() function)
                                                y_utm[~y_utm.mask].ravel()),                # valid (not masked) UTM Y coordinates of the sun angle values in vector format (.ravel() function)
                                               zenith_array[~zenith_array.mask].ravel(),    # valid sun angle values in vector format (.ravel() function)
                                               (x_im, y_im),                                # UTM X and Y coordinates of the pixels where to compute sun angles
                                               method='cubic')                              # method of the interpolation (possible options {‘linear’, ‘nearest’, ‘cubic’}, 'linear' is default)
    
    # The output interpolated array is in the vector format, so we reshape it to the mask format and multiply by mask to mask non-desired pixels with numpy.nan.
    # "zenith_array" is the final matrix with interpolated angles.
    zenith_array        = np.reshape(couche_interp, mask.shape)*mask

    # ==========================================================================
    # Sun Azimuth angles
    # ==========================================================================

    # The exact same thing is done for the Azimuth angles.
    azimuth_array        = read_grid_as_array(SunAzimuthNode[2])
    azimuth_array        = np.ma.array(azimuth_array, mask=np.isnan(azimuth_array))
    res_x_im, res_y_im  = geotransform[1], geotransform[5]
    x_im, y_im          = np.meshgrid(np.arange(coin_X+res_x_im/2, coin_X + mask.shape[1]*res_x_im,res_x_im),
                                      np.arange(coin_Y+res_y_im/2, coin_Y + mask.shape[0]*res_y_im,res_y_im))
    x_utm,y_utm         = np.meshgrid(np.arange(ulx_tile, ulx_tile + azimuth_array.shape[1]*res_x_angles,res_x_angles),
                                      np.arange(uly_tile, uly_tile + azimuth_array.shape[0]*res_y_angles,res_y_angles))
    x_utm               = np.ma.array(x_utm, mask=azimuth_array.mask)
    y_utm               = np.ma.array(y_utm, mask=azimuth_array.mask)
    couche_interp       = interpolate.griddata((x_utm[~x_utm.mask].ravel(),
                                                y_utm[~y_utm.mask].ravel()),
                                               azimuth_array[~azimuth_array.mask].ravel(),
                                               (x_im, y_im),
                                               method='cubic')
    azimuth_array        = np.reshape(couche_interp, mask.shape)*mask
    


def get_view_angle_information(angles_xml, detectors_mask, coin_X, coin_Y, mask, geotransform, band_id):
    '''
    Interpolate Sentinel-2 viewing zenith and azimuth angles to the band_id native spatial resolution
    using detectors footprints mask.

    Arguments:
        angles_xml      : (string) path to the MTD_TL.xml file which contains values of the sun angles.
        detectors_mask  : numpy.array with values corresponding to the detector ID, read from jp2 or gml files, and
                            cropped to the ROI pixels and shape (and multiplid by the mask to mask pixels that we don't want in our ROI)
        coin_X          : (number) UTM X coordinates of the upper left corner of te desired ROI to extract from the entire tile.
        coin_Y          : (number) UTM Y coordinates of the upper left corner of te desired ROI to extract from the entire tile.
        mask            : (numpy.array) array of the same shape as the ROI;
                            value 1 means that the coresponding pixel is in the ROI;
                            value numpy.nan means that the pixels is outside of the ROI;
                            this array is used to mask values outside of the ROI after the interpolation.
                            If you don't want to use this "feature" you can pass -1 for this argument.
        geotransform    : gdal GeoTransform variable (from gdal GetGeoTransform() function) which holds the geotransform of the jp2 S2 image with desired resolution
        band_id         : (int) index of the desired S2 band, starting at 0 (Band 1 -> band_id=0, Band 2 -> band_id=1 etc)

    Output:
        vza, vaa : numpy.arrays with the interpolated values for the given ROI
    '''

    # Get the UTM coordinates of the upper left corner of the Sun Angle matrix (not the S2 image tile) for the desired band through band_id
    root            = ElementTree().parse(angles_xml)
    band_nodes      = root.findall('.//Viewing_Incidence_Angles_Grids[@bandId="%d"]' %band_id)
    node            = root.find('.//Geoposition[@resolution="%d"]' %geotransform[1])
    ulx_tile        = float(node[0].text)
    uly_tile        = float(node[1].text)

    # Read Viewing Zenith and Azimuth angles values into numpy.arrays for each detector
    zenith_list, azimuth_list = [], []
    detectorIds = []
    for band_node in band_nodes:
        detectorIds.append(band_node.attrib['detectorId'])
        zenith_list.append(read_grid_as_array(band_node[0][2]))
        azimuth_list.append(read_grid_as_array(band_node[1][2]))
    if len(detectorIds)==0: # in case if there is no information about Viewing angles
        vza = np.full(mask.shape,np.nan)
        vaa = np.full(mask.shape,np.nan)
        return vza, vaa

    # print('detectorIds from xml -> ',detectorIds)
    
    # Spatial resolution of the Sun angle data
    res_x_angles = 5000
    if geotransform[5]>0:
        res_y_angles = 5000
    else:
        res_y_angles = -5000
    
    # Compute UTM coordinates for each pixel of the ROI
    res_x_im, res_y_im  = geotransform[1], geotransform[5]
    x_im, y_im          = np.meshgrid(np.arange(coin_X+res_x_im/2, coin_X + mask.shape[1]*res_x_im,res_x_im),
                                      np.arange(coin_Y+res_y_im/2, coin_Y + mask.shape[0]*res_y_im,res_y_im))
    
    # ==========================================================================
    # Viewing Zenith angles
    # ==========================================================================
    
    angles = []
    for ii, couche in enumerate(zenith_list): # first thing, we interpolate angles for every detector alone
        zenith_array = couche
        
        # Comput UTM coordinates for every Viewing angle value
        # -> si les coordonnées des angles sont aux centres des pixels
        # x_utm,y_utm = np.meshgrid(np.arange(ulx+res_x_angles/2, ulx + zenith_array.shape[1]*res_x_angles,res_x_angles),
        #                           np.arange(uly+res_y_angles/2, uly + zenith_array.shape[0]*res_y_angles,res_y_angles))
        # -> si les coordonnées des angles sont dans les coins hauts gauches des pixels
        x_utm,y_utm = np.meshgrid(np.arange(ulx_tile, ulx_tile + zenith_array.shape[1]*res_x_angles,res_x_angles),
                                  np.arange(uly_tile, uly_tile + zenith_array.shape[0]*res_y_angles,res_y_angles))
        
        # Mask NaNs in the original array
        zenith_array = np.ma.array(zenith_array, mask=np.isnan(zenith_array))

        # If there are more than 4 valid values in the array, we do the interpolation.
        # Used python interpolation requires more than 4 points. Usually ararys with less
        # than 4 points cover pixels that are already present in another detector
        # footprint for the same tile, so no information is lost.
        if zenith_array.count()>4:
            # Mask coordinates of the masked angle values
            x_utm = np.ma.array(x_utm, mask=zenith_array.mask)
            y_utm = np.ma.array(y_utm, mask=zenith_array.mask)
            # x_min, x_max = np.amin(x_utm), np.amax(x_utm)
            # y_min, y_max = np.amin(y_utm), np.amax(y_utm)

            # Interpolate valid values for the UTM corodinates of the image pixels
            # (same method that in the Sun angles interpolation)
            couche_interp = interpolate.griddata((x_utm[~x_utm.mask].ravel(),
                                                  y_utm[~y_utm.mask].ravel()),
                                                  zenith_array[~zenith_array.mask].ravel(),
                                                  (x_im, y_im),
                                                  method='cubic')

            # "Angles" is the list where each element is an array of interpolated angles for corresponding detector
            angles.append(couche_interp)

    # The last step is to merge arrays of each detector into one image

    # Find IDs of detectors present in the tile
    det_values      = set(detectors_mask.ravel())
    # print('unique det values from det mask -> ', det_values)

    # Prepare the array which will contain merged values of angles
    angles_matrix   = np.full(detectors_mask.shape,np.nan)
    for detector in det_values:
        # find which of the layer in the interpolated angles corresponds to the current detector
        try : index = detectorIds.index(str(detector))
        except : continue # if value not found then we this detector and its pixels will be NaNs
        # if value found, take the corresponding array("index") and the pixel covered by the current detector ("detectors_mask==detector")
        arr         = angles[index][detectors_mask==detector]
        # Take corresponding coordinates pixels
        x,y         = x_im[detectors_mask==detector], y_im[detectors_mask==detector]
        # If there is still NaN values, we interpolate them with nearest neighbour inside of the already interpolated angles
        # (same interpolation function and method as before)
        if np.any(np.isnan(arr)):
            try:
                arr_interp  = interpolate.griddata((x[~np.isnan(arr)],
                                                    y[~np.isnan(arr)]),
                                                   arr[~np.isnan(arr)],
                                                   (x[np.isnan(arr)],
                                                    y[np.isnan(arr)]),
                                                   method='nearest')
                arr[np.isnan(arr)] = arr_interp
            # If number of values is not sufficient
            except: arr[np.isnan(arr)] = -999

        # Replace pixels in the final array with the valeus of corresponding detector
        angles_matrix[detectors_mask==detector] = arr

    # Mask pixels with mask if needed
    vza = angles_matrix*mask
    
    # ==========================================================================
    # Viewing Azimuth angles
    # ==========================================================================
    
    # Same method/code as for zenith angles, but in the beginning we are reading the variable
    # "azimuth_list" where we have arrays of viewing azimuth angle values

    angles = []
    for ii, couche in enumerate(azimuth_list):
        zenith_array = couche

        # -> si les coordonnées des angles sont aux centres des pixels
        # x_utm,y_utm = np.meshgrid(np.arange(ulx+res_x_angles/2, ulx + zenith_array.shape[1]*res_x_angles,res_x_angles),
        #                           np.arange(uly+res_y_angles/2, uly + zenith_array.shape[0]*res_y_angles,res_y_angles))
        # -> si les coordonnées des angles sont dans les coins hauts gauches des pixels
        x_utm,y_utm = np.meshgrid(np.arange(ulx_tile, ulx_tile + zenith_list[0].shape[1]*res_x_angles,res_x_angles),
                                  np.arange(uly_tile, uly_tile + zenith_list[0].shape[0]*res_y_angles,res_y_angles))

        zenith_array = np.ma.array(zenith_array, mask=np.isnan(zenith_array))

        if zenith_array.count()>4:
            x_utm = np.ma.array(x_utm, mask=zenith_array.mask)
            y_utm = np.ma.array(y_utm, mask=zenith_array.mask)
            # x_min, x_max = np.amin(x_utm), np.amax(x_utm)
            # y_min, y_max = np.amin(y_utm), np.amax(y_utm)

            couche_interp = interpolate.griddata((x_utm[~x_utm.mask].ravel(),
                                                  y_utm[~y_utm.mask].ravel()),
                                                  zenith_array[~zenith_array.mask].ravel(),
                                                  (x_im, y_im),method='cubic')

            angles.append(couche_interp)

    angles_matrix   = np.full(detectors_mask.shape,np.nan)
    for detector in det_values:
        try : index = detectorIds.index(str(detector))
        except : continue
        arr         = angles[index][detectors_mask==detector]
        x,y         = x_im[detectors_mask==detector], y_im[detectors_mask==detector]
        if np.any(np.isnan(arr)):
            try:
                arr_interp  = interpolate.griddata((x[~np.isnan(arr)],
                                                    y[~np.isnan(arr)]),
                                                   arr[~np.isnan(arr)],
                                                   (x[np.isnan(arr)],
                                                    y[np.isnan(arr)]),
                                                   method='nearest')
                arr[np.isnan(arr)] = arr_interp
            except: arr[np.isnan(arr)] = -999

        angles_matrix[detectors_mask==detector] = arr

    vaa = angles_matrix*mask

    return vza, vaa