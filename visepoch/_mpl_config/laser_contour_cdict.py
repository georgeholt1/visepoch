# Author: George K. Holt
# License: MIT
# Version: 0.1.0
"""
Part of VISFBPIC.

Custom colour map for laser contours.
"""
from matplotlib.colors import LinearSegmentedColormap
cdict = {
    'red': (
        (0.0, 0.86, 0.86),
        (1.0, 0.86, 0.86)
    ),
    'green': (
        (0.0, 0.37, 0.37),
        (1.0, 0.37, 0.37)
    ),
    'blue': (
        (0.0, 0.34, 0.34),
        (1.0, 0.34, 0.34)
    ),
    'alpha': (
        (0.0, 0.0, 0.0),
        (1.0, 0.5, 0.5)
    )
}
laser_cmap = LinearSegmentedColormap('custom_cmap', cdict)