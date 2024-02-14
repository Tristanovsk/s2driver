# s2driver package
## Tool for easy loading of Sentinel-2 L1C SAFE format with accurate computation of the viewing angles

# Installation on TREX
module load conda
conda env create -f environment.yml
conda activate grs_cnes
pip install .

### Usual installation, please use conda environment
conda create -n "YOUR_ENV" python=3.11

conda activate "YOUR_ENV"

conda install gdal numba rasterio

pip install .

## Example

![example gif](illustration/s2driver_visual_tool_optimized.gif)


## 2D-fiiting method for angle computation

![example files](fig/example_3D_fitting_one_detector_v2.png)
![example files](fig/example_2D_fitting_one_band_v3.png)
![example files](fig/example_scattering_angle_all_bands.png)
![example files](fig/example_reflectance_all_bands.png)
![example files](fig/example_ndwi_mask.png)
