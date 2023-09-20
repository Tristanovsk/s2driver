'''
v1.0.2: fix for EOreader version >= 0.23
v1.0.3: add landsat driver for L8 and L9
'''

__version__='1.0.3'

from .driver_landsat_col2 import landsat_driver
from .driver_S2_SAFE import sentinel2_driver