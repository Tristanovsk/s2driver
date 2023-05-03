# s2driver package
## Tool for easy loading of Sentinel-2 L1C SAFE format with accurate computation of the viewing angles

# Installation
### please use conda environment
conda activate "name of your conda environment"
conda install numba rasterio
conda install gdal

python setup.py install

## Installation of the missing modules

conda install "name of the missing module"

## If it doesn't work because of conflicts between modules versions try :

pip install "name of the missing module"

## Example

![example gif](illustration/s2driver_visual_tool_optimized.gif)


## 2D-fiiting method for angle computation

![example files](fig/example_3D_fitting_one_detector_v2.png)
![example files](fig/example_2D_fitting_one_band_v3.png)
![example files](fig/example_scattering_angle_all_bands.png)
![example files](fig/example_reflectance_all_bands.png)
![example files](fig/example_ndwi_mask.png)
