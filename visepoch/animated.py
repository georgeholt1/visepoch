# Author: George K. Holt
# License: MIT
# Version: 0.0.1
"""
Part of VISEPOCH.

Contains methods for creating animations of the EPOCH data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation

import sdf

from visepoch._mpl_config.laser_contour_cdict import laser_cmap
from visepoch._mpl_config.cmaps import haline_cmap



plt.style.use(os.path.join(
    os.path.split(__file__)[0], '_mpl_config', 'style.mplstyle'))



def _find_nearest_idx(a, v):
    '''Return index of array a with value closest to v.'''
    a = np.asarray(a)
    return (np.abs(a - v)).argmin()



def animated_plasma_density(
    sup_dir,
    out_dir=None,
    x_roi=(0, None),
    y_roi=None,
    n_max=None,
    relative_n_max=None,
    show_laser=False,
    dpi=200,
    figsize=(8, 4.5),
    cmap="haline",
    x_units="window",
    interval=100
):
    '''Create animated number density plot.
    
    Parameters
    ----------
    sup_dir : str
        Path to the simulation super directory. The simulations diags should be
        in here.
    out_dir : str, optional
        Path to output directory within which to save the animation. Defaults to
        None, which saves to <sup_dir>/analysis/plasma_density/
    x_roi : tuple, optional
        Size 2 tuple with x-directional coordinates of the plotting region in
        relative units (x_min, x_max). 0 is the x_min boundary. 1 is the x_max
        boundary. Defaults to (0, None), which selects the whole domain.
    y_roi : float, optional
        The data is cropped in the y-direction to +/- y_roi in SI units.
        Defaults to None, which performs no cropping.
    n_max : float, optional
        The maximum value of the colour scale. Either this or relative_n_max
        must be supplied. Defaults to None.
    relative_n_max : float, optional
        The maximum value of the colour scale relative to the maximum number
        density at any time in the simulation. Either this or n_max must be
        supplied. Defaults to None.
    show_laser : bool, optional
        Whether or not to plot the laser envelope derived from the Poynting
        flux. Defaults to False.
    dpi : int, optional
        Dots per inch resolution. Changing this parameter may result in a bad
        plot layout. Defaults to 200.
    figsize : tuple, optional
        Figure size in the form (width, height). Changing this parameter may
        result in a bad plot layout. Defaults to (8, 4.5).
    cmap : str, optional
        Colour map for the number density plot. Either "haline" or "viridis".
        Defaults to "haline".
    x_units : str, optional
        Units of the x-axis. Either 'window' (default) or 'simulation'.
    interval : int, optional
        Interval between frames in ms. Governs the length of the animation.
        Defaults to 100.
        
    Input deck requirements
    -----------------------
    In addition to base requirements:
        - number_density.
        - poynting_flux if show_laser is True.
    '''
    if n_max is None and relative_n_max is None:
        raise ValueError("Either n_max or relative_n_max must be supplied")
    elif n_max is not None and relative_n_max is not None:
        print("n_max and relative_n_max both supplied, defaulting to n_max")
        c_scale = "absolute"
    elif n_max is None and relative_n_max is not None:
        c_scale = "relative"
    elif n_max is not None and relative_n_max is None:
        c_scale = "absolute"
        
    if out_dir is None:
        out_dir = os.path.join(sup_dir, "analysis", "plasma_density")
        print("Output directory not specified")
        print(f"Defaulting to {out_dir}")
    if not os.path.isdir(out_dir):
        print(f"Making output directory at {out_dir}")
        os.makedirs(out_dir)
        
    # get and order sdf files
    sim_files = []
    for file in os.listdir(sup_dir):
        if file.endswith('.sdf'):
            sim_files.append(file)
    sim_files.sort()
    
    # get maximum number density in simulation
    n_e_max = 0
    for f in sim_files:
        data = sdf.read(os.path.join(sup_dir, f))
        if data.Derived_Number_Density_electron.data.max() > n_e_max:
            n_e_max = data.Derived_Number_Density_electron.data.max()
            
    # get maximum number density for plot
    if c_scale == "relative":
        n_e_max_plot = n_e_max * relative_n_max
    elif c_scale == "absolute":
        n_e_max_plot = n_max
    n_e_max_plot_oom = int(np.floor(np.log10(n_e_max_plot)))
    n_e_max_plot /= 10.0 ** n_e_max_plot_oom
    
    # get initial info
    data = sdf.read(os.path.join(sup_dir, sim_files[0]))
    n_e = data.Derived_Number_Density_electron.data.T
    grid = data.Grid_Grid_mid.data
    if show_laser:
        Px = data.Derived_Poynting_Flux_x.data.T
    
    # get indices of ROI
    if y_roi is None:
        y_low_idx = 0
        y_up_idx = None
    else:
        y_low_idx = _find_nearest_idx(grid[1], -y_roi)
        y_up_idx = _find_nearest_idx(grid[1], y_roi)
    if x_roi[0] == 0:
        x_low_idx = 0
    else:
        x_low_idx = int(grid[0].size * x_roi[0])
    if x_roi[1] is None:
        x_up_idx = None
    else:
        x_up_idx = int(grid[0].size * x_roi[1])
    
    # initial image extent
    extent = np.array([0.0, 0.0, 0.0, 0.0])
    if x_units == "window":
        extent[0] = 0.0
        if x_up_idx is None:
            extent[1] = grid[0][-1] - grid[0][x_low_idx]
        else:
            extent[1] = grid[0][x_up_idx] - grid[0][x_low_idx]
    else:
        extent[0] = grid[0][x_low_idx]
        if x_up_idx is None:
            extent[1] = grid[0][-1]
        else:
            extent[1] = grid[0][x_up_idx]
    extent[2] = grid[1][y_low_idx]
    if y_up_idx is None:
        extent[3] = grid[1][-1]
    else:
        extent[3] = grid[1][y_up_idx]
    
    # set up figure
    fig = plt.figure(
        constrained_layout=False,
        figsize=figsize,
        dpi=dpi
    )
    spec = gridspec.GridSpec(
        ncols=2, nrows=1,
        left=0.1, right=0.9,
        bottom=0.1, top=0.9,
        wspace=0.05,
        width_ratios=[1, 0.03]
    )
    ax = fig.add_subplot(spec[0])
    cax = fig.add_subplot(spec[1])
    
    # get colormap
    if cmap == "haline":
        cmap = haline_cmap
    elif cmap == "viridis":
        cmap = "viridis"
    else:
        raise ValueError("cmap must be haline or viridis")
    
    # draw first frame
    im = ax.imshow(
        np.divide(
            n_e[y_low_idx:y_up_idx, x_low_idx:x_up_idx],
            10.0**n_e_max_plot_oom,
            dtype=np.float64
        ),
        aspect='auto',
        extent=extent*1e6,  # microns
        vmin=0,
        vmax=n_e_max_plot,
        cmap=cmap
    )
    if show_laser:
        if x_units == "window":
            cs_x = grid[0][x_low_idx:x_up_idx] - grid[0][x_low_idx]
        else:
            cs_x = grid[0][x_low_idx:x_up_idx]
        print(Px[y_low_idx:y_up_idx, x_low_idx:x_up_idx].max())
        cs = [ax.contour(
            cs_x*1e6,  # microns
            grid[1][y_low_idx:y_up_idx]*1e6,  # microns
            Px[y_low_idx:y_up_idx, x_low_idx:x_up_idx],
            levels=Px.max()*np.linspace(1/np.e**2, 0.95, 4),
            cmap=laser_cmap
        )]
    fig.colorbar(im, cax=cax, extend='max')
    title_text = fig.text(
        0.5, 0.95,
        r'$t = {:.1f}$ ps'.format(data.Header['time']*1e12),
        va='center', ha='center', fontsize=11
    )
    ax.set_ylabel('$y$ ($\mathrm{\mu}$m)')
    if x_units == "window":
        ax.set_xlabel("$\zeta$ ($\mathrm{\mu}$m)")
    else:
        ax.set_xlabel("$x$ ($\mathrm{\mu}$m)")
    cax.set_ylabel(
        r'$n_e$ ($\times 10 ^ {{{}}}$ '.format(n_e_max_plot_oom) + 'm$^{-3}$)'
    )
    
    print("Animating")
    
    def animate_plasma_density(i):
        '''Update the plot.'''
        print_string = str(i+1) + ' / ' + str(len(sim_files))
        print(print_string.ljust(20), end='\r', flush=True)
        
        # get data
        data = sdf.read(os.path.join(sup_dir, sim_files[i]))
        n_e = data.Derived_Number_Density_electron.data.T
        grid = data.Grid_Grid_mid.data
        if show_laser:
            Px = data.Derived_Poynting_Flux_x.data.T
        
        # redraw
        im.set_data(
            np.divide(
                n_e[y_low_idx:y_up_idx, x_low_idx:x_up_idx],
                10.0**n_e_max_plot_oom,
                dtype=np.float64
            )
        )
        if x_units == "simulation":
            extent[0] = grid[0][x_low_idx]
            if x_up_idx is None:
                extent[1] = grid[0][-1]
            else:
                extent[1] = grid[0][x_up_idx]
            im.set_extent(extent*1e6)
        if show_laser:
            for c in cs[0].collections:
                c.remove()
            if x_units == "window":
                cs_x = grid[0][x_low_idx:x_up_idx] - grid[0][x_low_idx]
            else:
                cs_x = grid[0][x_low_idx:x_up_idx]
            cs[0] = ax.contour(
                cs_x*1e6,
                grid[1][y_low_idx:y_up_idx]*1e6,
                Px[y_low_idx:y_up_idx, x_low_idx:x_up_idx],
                levels=Px.max()*np.linspace(1/np.e**2, 0.95, 4),
                cmap=laser_cmap
            )
            
        # need to do this twice because of contours
        if x_units == "simulation":
            im.set_extent(extent*1e6)
            
        title_text.set_text(r'$t = {:.1f}$ ps'.format(data.Header['time']*1e12))
        
    ani = animation.FuncAnimation(
        fig, animate_plasma_density, range(0, len(sim_files)),
        repeat=False, blit=False, interval=interval
    )
    
    ani.save(os.path.join(out_dir, "number_density_laser.mp4"))
            