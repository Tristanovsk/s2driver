'''
v1.0.2: fix for EOreader version >= 0.23
v1.0.3: add landsat driver for L8 and L9
v1.0.4: add subset option for S2
'''

__version__='1.0.4'

from .driver_landsat_col2 import landsat_driver
from .driver_S2_SAFE import sentinel2_driver