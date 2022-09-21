import os, sys #; sys.path.append('../')

# Prettymaps
from prettymaps import *
# Vsketch
#import vsketch
# OSMNX
import osmnx as ox
ox.config(use_cache=True, log_console=True)
# Matplotlib-related
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from descartes import PolygonPatch
# Shapely
from shapely.geometry import *
from shapely.affinity import *
from shapely.ops import unary_union
plt.ioff()
fig, ax = plt.subplots(figsize = (3*3000/300, 3000/300), constrained_layout = True, dpi = 300)
fig.patch.set_facecolor('#FFEDDF')
point = (43.35,3.61)
layers = plot(
     point,
     # The coordinates of the center of the circle , Format ：( latitude , longitude )
    radius = 15000, # radius


    # Matplotlib 'ax'
    ax = ax,

    # Layers to plot & their kwargs
    layers = {
        'perimeter': {},
        'streets':      {
            'width': {
                'motorway':     12,
                'trunk':        12,
                'primary':      11,
                'secondary':    10,
                'tertiary':     9,
                'residential':  8,
            }
        },
        'park':         {'tags': {'leisure': 'park', 'landuse': 'golf_course', 'landuse': 'meadow', 'leisure': 'nature_reserve', 'boundary': 'protected_area', 'place': 'square', 'natural': 'grassland', 'landuse': 'military', 'amenity': 'hospital'}},
        'grass':        {'tags': {'landuse': 'grass', 'natural': 'wood'}},
        'wetland':      {'tags': {'natural': 'wetland', 'natural': 'scrub'}},
        'beach':        {'tags': {'natural': 'beach'}},
        'water':        {'tags': {'natural': 'water'}},
        'pedestrian':   {'tags': {'area:highway': 'pedestrian'}},
        'building':     {'tags': {'building': True}}
    },
    drawing_kwargs = {
        'perimeter':    {'ec': '#0F110C', 'fill': False, 'lw': 0},
        'park':         {'fc': '#AAD897', 'ec': '#8bc49e', 'lw': 0, 'zorder': 1, 'hatch': 'ooo...'},
        'grass':        {'fc': '#72C07A', 'ec': '#64a38d', 'lw': 0, 'zorder': 1, 'hatch': 'ooo...'},
        'wetland':      {'fc': '#D2D68D', 'ec': '#AEB441', 'lw': 0, 'zorder': 3, 'hatch': 'ooo...'},
        'water':        {'fc': '#6CCFF6', 'ec': '#59adcf', 'lw': 0, 'zorder': 2, 'hatch': 'ooo...'},
        'beach':        {'fc': '#F2E3BC', 'ec': '#EBD499', 'lw': 0, 'zorder': 2, 'hatch': 'ooo...'},
        'pedestrian':   {'fc': '#7BC950', 'ec': '#638475', 'lw': 0, 'zorder': 2, 'hatch': 'ooo...'},
        'streets':      {'fc': '#898989', 'ec': '#706f6f', 'zorder': 3, 'lw': 0, 'hatch': 'ooo...'},
        'building':     {'fc': '#E7A89C', 'ec': '#E7A89C', 'lw': 0, 'zorder': 0},
    },

    osm_credit = {'x': -.06, 'y': -.05, 'color': '#706f6f'}
)

# Add meadows, parks & scrubs
for tags, kwargs in [
    ({'landuse': 'meadow'}, {'fc': '#AAD897', 'ec': '#8bc49e', 'lw': 0, 'zorder': 1, 'hatch': 'ooo...'}),
    ({'leisure': 'park'}, {'fc': '#AAD897', 'ec': '#8bc49e', 'lw': 0, 'zorder': 1, 'hatch': 'ooo...'}),
    ({'natural': 'scrub'}, {'fc': '#D2D68D', 'ec': '#AEB441', 'lw': 0, 'zorder': 3, 'hatch': 'ooo...'}),
]:
    ax.add_patch(PolygonPatch(
        unary_union(
            ox.project_gdf(
                ox.geometries_from_point(
                    point,
                    tags = tags,
                    dist = 1000
                )
            ).geometry
        ),
        **kwargs
    ))

# Add 'sea'
sea = max(layers['perimeter'].convex_hull.difference(layers['perimeter']), key = lambda x: x.area).buffer(20)
sea = sea.difference(translate(scale(sea, 1.05, 1), 0, -200)).difference(layers['perimeter'])[0]
ax.add_patch(PolygonPatch(sea, fc = '#59A5D8', ec = '#386FA4', hatch = 'ooo...'))

# Set bounds
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
dx = xmax-xmin
dy = ymax-ymin
ax.set_xlim(xmin+.3*dx, xmax-.3*dx)
ax.set_ylim(ymin+.3*dy, ymax-.0*dy)

dir = '/DATA/projet/magellium/malaigue/vrac'
file = os.path.join(dir,'thau_map2')
plt.savefig(file+'.png')
plt.savefig(file+'.svg')