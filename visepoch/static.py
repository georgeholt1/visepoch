# Author: George K. Holt
# License: MIT
# Version: 0.1.0
"""
Part of VISEPOCH.

Contains functions for creating static plots from the EPOCH data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c, epsilon_0
from scipy.interpolate import CubicSpline
from tqdm import tqdm

import sdf



plt.style.use(os.path.join(
    os.path.split(__file__)[0], '_mpl_config', 'style.mplstyle'))



def _find_nearest_idx(a, v):
    '''Return index of array a with value closest to v.'''
    a = np.asarray(a)
    return (np.abs(a - v)).argmin()



def _weighted_avg_std(arr, w):
    '''Calculated weighted mean and standard deviation of array.
    
    Parameters
    ----------
    arr : numpy ndarray
        Array of values.
    w : numpy ndarray
        Array of weights corresponding to arr.
    '''
    avg = np.average(arr, weights=w)           # average
    var = np.average((arr-avg)**2, weights=w)  # variance
    std = np.sqrt(var)                         # standard deviation
    return (avg, std)



def plot_average_electron_energy(
    sup_dir,
    out_dir=None,
    x_select=(None, None),
    figsize=(8, 4.5),
    dpi=200,
):
    '''Plot average and standard deviation of energy of electrons in subset.
    
    Units of energy are converted to eV for plotting.
    
    Parameters
    ----------
    sup_dir : str
        Path to the simulation super directory. The simulation dumps shouyld be
        in here.
    out_dir : str, optional
        Path to output directory within which to save the plot. Defaults to
        None, which saves to <sup_dir>/analysis/average_electron_energy/.
    x_select : tuple, optional
        2-tuple with float values specifying the x-position of particles to
        sub-sample in SI relative window units (xmin, xmax). A value of None
        sets the sub-sample to be the minimum of maximum x-border.
        E.g. (5e-6, 20e-6) will sample the particles with position values
        between 5 and 20 microns in the window units.
        Defaults to (None, None), which performs no sub-sampling.
    figsize : tuple, optional
        Figure size in the form (width, height). Changing this parameter may
        result in a bad plot layout. Defaults to (8, 4.5).
    dpi : int, optional
        Dots per inch resolution. Changing this parameter may result in a bad
        plot layout. Defaults to 200.
        
    Input deck requirements
    -----------------------
    In addition to base requirements:
        - A subset of the `electron` species called `high_gamma` (example
          below).
        - Particle positions, weights and energies dumped for the `high_gamma`
          subset.
    ```
    begin:subset
        name = high_gamma
        gamma_min = 50
        include_species:electron
    end:subset
    ```
    '''
    # make output directory
    if out_dir is None:
        out_dir = os.path.join(sup_dir, "analysis", "average_electron_energy")
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
    
    # initialise lists for storing values
    t = np.zeros(
        (len(sim_files), ),
        dtype=np.float64
    )  # dump time stamps
    Ek_avg = np.zeros(
        (len(sim_files), ),
        dtype=np.float64
    )  # energy weighted mean
    Ek_std = np.zeros(
        (len(sim_files), ),
        dtype=np.float64
    )  # energy weighted standard deviation
    
    # loop over data files to perform measurements
    for i, f in enumerate(sim_files):
        data = sdf.read(os.path.join(sup_dir, f))
        # if AttributeError occurs that means there are no particles in the
        #   subset
        try:
            Ek_temp = data.Particles_Ek_subset_high_gamma_electron.data
            w_temp = data.Particles_Weight_subset_high_gamma_electron.data
            x_temp = data.Grid_Particles_subset_high_gamma_electron.data[0]
            sub_sample = True
        except AttributeError:
            Ek_temp = 0
            w_temp = 1
            sub_sample = False
        
        # perform position-based sub-sampling if there are particles
        if sub_sample:
            # get x region for sub-sampling
            grid = data.Grid_Grid.data
            if x_select[0] is not None:
                x_min_sample = grid[0].min() + x_select[0]
            else:
                x_min_sample = grid[0].min()
            if x_select[1] is not None:
                x_max_sample = grid[0].min() + x_select[1]
            else:
                x_max_sample = grid[0].max()
                
            # check sub-sample region is valid
            if x_min_sample >= x_max_sample:
                raise ValueError("Invalid x_select")
            
            # perform sub-sample
            arr_inds = np.where(
                (x_temp<=x_max_sample) & (x_temp >= x_min_sample)
            )
            Ek_temp = Ek_temp[arr_inds]
            w_temp = w_temp[arr_inds]
            
            # check that there are still particles
            if Ek_temp.size == 0:
                Ek_temp = 0
                w_temp = 1
            
        
        Ek_avg_temp, Ek_std_temp = _weighted_avg_std(Ek_temp, w_temp)
        
        t[i] = data.Header['time']
        Ek_avg[i] = Ek_avg_temp
        Ek_std[i] = Ek_std_temp
        
    # check there is data to plot
    if Ek_avg.max() == 0:
        print("No electrons found")
        return
    
    # convert to eV
    Ek_avg_eV = Ek_avg / 1.609e-19
    Ek_std_eV = Ek_std / 1.609e-19
    
    # calculate some orders of magnitude    
    t_max_oom = int(np.floor(np.log10(t.max())))
    Ek_max_plot = np.add(Ek_avg_eV, Ek_std_eV).max() * 1.1
    Ek_max_plot_oom = int(np.floor(np.log10(Ek_max_plot)))
    Ek_max_plot /= 10.0 ** Ek_max_plot_oom

    # plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    ax.fill_between(
        t/10.0**t_max_oom,
        Ek_avg_eV/10.0**Ek_max_plot_oom-Ek_std_eV/10.0**Ek_max_plot_oom,
        Ek_avg_eV/10.0**Ek_max_plot_oom+Ek_std_eV/10.0**Ek_max_plot_oom,
        alpha=0.3
    )
    ax.plot(
        t/10.0**t_max_oom,
        Ek_avg_eV/10.0**Ek_max_plot_oom
    )
    ax.grid()
    ax.set_xlim(t.min()/10.0**t_max_oom, t.max()/10.0**t_max_oom)
    ax.set_ylim(0, Ek_max_plot)
    ax.set_xlabel(r'$t$ ($\times$' + f'$10 ^ {{{t_max_oom}}}$' + ' s)')
    ax.set_ylabel(r'$E_k$ ($\times$' + f'$10 ^ {{{Ek_max_plot_oom}}}$' + ' eV)')
    
    fig.tight_layout()
    
    fig.savefig(
        os.path.join(out_dir, "average_electron_energy.png"),
        dpi=dpi
    )
    
    plt.close()
    
    
    
def plot_charge(
    sup_dir,
    out_dir=None,
    x_select=(None, None),
    figsize=(8, 4.5),
    dpi=200
):
    '''Plot total charge of electron subset.
    
    Parameters
    ----------
    sup_dir : str
        Path to the simulation super directory. The simulation dumps shouyld be
        in here.
    out_dir : str, optional
        Path to output directory within which to save the plot. Defaults to
        None, which saves to <sup_dir>/analysis/average_electron_energy/.
    x_select : tuple, optional
        2-tuple with float values specifying the x-position of particles to
        sub-sample in SI relative window units (xmin, xmax). A value of None
        sets the sub-sample to be the minimum of maximum x-border.
        E.g. (5e-6, 20e-6) will sample the particles with position values
        between 5 and 20 microns in the window units.
        Defaults to (None, None), which performs no sub-sampling.
    figsize : tuple, optional
        Figure size in the form (width, height). Changing this parameter may
        result in a bad plot layout. Defaults to (8, 4.5).
    dpi : int, optional
        Dots per inch resolution. Changing this parameter may result in a bad
        plot layout. Defaults to 200.
        
    Input deck requirements
    -----------------------
    In addition to base requirements:
        - A subset of the `electron` species called `high_gamma` (example
          below).
        - Particle positions and weights dumped for the `high_gamma` subset.
    ```
    begin:subset
        name = high_gamma
        gamma_min = 50
        include_species:electron
    end:subset
    ```
    '''
    # make output directory
    if out_dir is None:
        out_dir = os.path.join(sup_dir, "analysis", "electron_charge")
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
    
    # initialise lists for storing values
    t = np.zeros(
        (len(sim_files), ),
        dtype=np.float64
    )  # dump time stamps
    q = np.zeros(
        (len(sim_files), ),
        dtype=np.float64
    )  # charge
    
    # loop over data files to perform measurements
    for i, f in enumerate(sim_files):
        data = sdf.read(os.path.join(sup_dir, f))
        # if AttributeError occurs that means there are no particles in the
        #   subset
        try:
            w_temp = data.Particles_Weight_subset_high_gamma_electron.data
            x_temp = data.Grid_Particles_subset_high_gamma_electron.data[0]
            sub_sample = True
        except AttributeError:
            w_temp = 0
            sub_sample = False
        
        # perform position-based sub-sampling if there are particles
        if sub_sample:
            # get x region for sub-sampling
            grid = data.Grid_Grid.data
            if x_select[0] is not None:
                x_min_sample = grid[0].min() + x_select[0]
            else:
                x_min_sample = grid[0].min()
            if x_select[1] is not None:
                x_max_sample = grid[0].min() + x_select[1]
            else:
                x_max_sample = grid[0].max()
                
            # check sub-sample region is valid
            if x_min_sample >= x_max_sample:
                raise ValueError("Invalid x_select")
            
            # perform sub-sample
            arr_inds = np.where(
                (x_temp<=x_max_sample) & (x_temp >= x_min_sample)
            )
            w_temp = w_temp[arr_inds]
            
            # check that there are still particles
            if w_temp.size == 0:
                w_temp = 0
                
        # convert sum of weights to charge
        q_temp = np.sum(w_temp) * e
        
        t[i] = data.Header['time']
        q[i] = q_temp
        
    # check there is data to plot
    if q.max() == 0:
        print("No electrons found")
        return
    
    # calculate some orders of magnitude
    t_max_oom = int(np.floor(np.log10(t.max())))
    q_max_plot = q.max() * 1.1
    q_max_plot_oom = int(np.floor(np.log10(q_max_plot)))
    q_max_plot /= 10.0 ** q_max_plot_oom

    # plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    ax.plot(
        t/10.0**t_max_oom,
        q/10.0**q_max_plot_oom
    )
    ax.grid()
    ax.set_xlim(t.min()/10.0**t_max_oom, t.max()/10.0**t_max_oom)
    ax.set_ylim(0, q_max_plot)
    ax.set_xlabel(r'$t$ ($\times$' + f'$10 ^ {{{t_max_oom}}}$' + ' s)')
    ax.set_ylabel(r'$q$ ($\times$' + f'$10 ^ {{{q_max_plot_oom}}}$' + ' C)')
    
    fig.tight_layout()
    
    fig.savefig(
        os.path.join(out_dir, "electron_charge.png"),
        dpi=dpi
    )
    
    plt.close()
    
    
    
def plot_a0(
    sup_dir,
    cpw_x,
    lambda0=800e-9,
    out_dir=None,
    figsize=(8, 4.5),
    dpi=200
):
    '''Plot dimensionless laser amplitude.
    
    Parameters
    ----------
    sup_dir : str
        Path to the simulation super directory. The simulation dumps shouyld be
        in here.
    cpw_x : int
        Cells per laser central wavelength in the x-direction. E.g. if laser
        wavelength is 800 nm and cell size if 40 nm, cpw_x = 20.
    lambda0 : float, optional
        Laser central wavelength in SI units. Defaults to 800e-9.
    out_dir : str, optional
        Path to output directory within which to save the plot. Defaults to
        None, which saves to <sup_dir>/analysis/average_electron_energy/.
    figsize : tuple, optional
        Figure size in the form (width, height). Changing this parameter may
        result in a bad plot layout. Defaults to (8, 4.5).
    dpi : int, optional
        Dots per inch resolution. Changing this parameter may result in a bad
        plot layout. Defaults to 200.
    
    Input deck requirements
    -----------------------
    In addition to base requirements:
        - poynting_flux.
    '''
    # make output directory
    if out_dir is None:
        out_dir = os.path.join(sup_dir, "analysis", "a0")
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
    
    # initialise lists for storing values
    t = np.zeros(
        (len(sim_files), ),
        dtype=np.float64
    )  # dump time stamps
    a0 = np.zeros(
        (len(sim_files), ),
        dtype=np.float64
    )  # laser a0
    
    # loop over data files to perform measurements
    for i, f in enumerate(sim_files):
        data = sdf.read(os.path.join(sup_dir, f))
        
        # lineout of x-directional Poynting flux
        s = data.Derived_Poynting_Flux_x.data.T.shape[0]
        Px = data.Derived_Poynting_Flux_x.data.T[s//2, :]
        
        # moving average with window size equal to cells per wavelength
        Px_ma = np.convolve(Px, np.ones(cpw_x), 'valid') / cpw_x

        # peak intensity
        I0 = Px_ma.max()
        
        # a0 = 0.855 * lambda0 [micron] * sqrt(I [10^18 W/cm^2])
        a0[i] = 0.855 * lambda0 * 1e6 * np.sqrt(I0 * 1e-22)
        
        t[i] = data.Header['time']
    
    # calculate some orders of magnitude    
    t_max_oom = int(np.floor(np.log10(t.max())))
    a0_max_plot = a0.max() * 1.1
    a0_max_plot_oom = int(np.floor(np.log10(a0_max_plot)))
    a0_max_plot /= 10.0 ** a0_max_plot_oom

    # plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    ax.plot(
        t/10.0**t_max_oom,
        a0/10.0**a0_max_plot_oom
    )
    ax.grid()
    ax.set_xlim(t.min()/10.0**t_max_oom, t.max()/10.0**t_max_oom)
    ax.set_ylim(0, a0_max_plot)
    ax.set_xlabel(r'$t$ ($\times$' + f'$10 ^ {{{t_max_oom}}}$' + ' s)')
    ax.set_ylabel(r'$a_0$ ($\times$' + f'$10 ^ {{{a0_max_plot_oom}}}$' + ')')
    
    fig.tight_layout()
    
    fig.savefig(
        os.path.join(out_dir, "a0.png"),
        dpi=dpi
    )
    
    plt.close()
    
    
    
def plot_electron_energy_spectrum(
    sup_dir,
    energy_bins=1000,
    energy_units='gamma',
    x_select=(None, None),
    out_dir=None,
    figsize=(8, 4.5),
    dpi=200,
):
    '''Plot the energy spectrum of electrons in the subset.
    
    Parameters
    ----------
    sup_dir : str
        Path to the simulation super directory. The simulation dumps shouyld be
        in here.
    energy_bins : int, optional
        Number of bins to form the spectrum in energy. Defaults to 1000.
    energy_units : str, optional
        Units of energy to plot. Either 'gamma' (Lorentz factor) or 'eV'
        (electron volts). Defaults to 'gamma'.
    x_select : tuple, optional
        2-tuple with float values specifying the x-position of particles to
        sub-sample in SI relative window units (xmin, xmax). A value of None
        sets the sub-sample to be the minimum of maximum x-border.
        E.g. (5e-6, 20e-6) will sample the particles with position values
        between 5 and 20 microns in the window units.
        Defaults to (None, None), which performs no sub-sampling.
    out_dir : str, optional
        Path to output directory within which to save the plot. Defaults to
        None, which saves to <sup_dir>/analysis/average_electron_energy/.
    figsize : tuple, optional
        Figure size in the form (width, height). Changing this parameter may
        result in a bad plot layout. Defaults to (8, 4.5).
    dpi : int, optional
        Dots per inch resolution. Changing this parameter may result in a bad
        plot layout. Defaults to 200.
        
    Input deck requirements
    -----------------------
    In addition to base requirements:
        - A subset of the `electron` species called `high_gamma` (example
          below).
        - Particle positions, weights and energies dumped for the `high_gamma`
          subset.
    '''
    # make output directory
    if out_dir is None:
        out_dir = os.path.join(sup_dir, "analysis", "electron_energy_spectrum")
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
    
    # initialise arrays
    t = np.zeros(
        (len(sim_files), ),
        dtype=np.float64
    )  # dump time stamps
    spectrum = np.zeros(
        (energy_bins, len(sim_files)),
        dtype=np.float64
    )  # the energy spectrum image-like
    
    # get maximum energy in simulation
    Ek_max = 0
    for f in sim_files:
        data = sdf.read(os.path.join(sup_dir, f))
        # if AttributeError occurs that means there are no particles in the
        #   subset
        try:
            Ek_temp = data.Particles_Ek_subset_high_gamma_electron.data.max()
        except AttributeError:
            Ek_temp = 0
        
        if Ek_temp > Ek_max:
            Ek_max = Ek_temp
    
    # histogram bin edges
    bin_edges = np.linspace(0, Ek_max, energy_bins+1)
    
    # populate the energy spectrum image-like
    for i, f in enumerate(sim_files):
        data = sdf.read(os.path.join(sup_dir, f))
        # if AttributeError occurs that means there are no particles in the
        #   subset
        try:
            Ek_temp = data.Particles_Ek_subset_high_gamma_electron.data
            w_temp = data.Particles_Weight_subset_high_gamma_electron.data
            x_temp = data.Grid_Particles_subset_high_gamma_electron.data[0]
            sub_sample = True
            fill = True
        except AttributeError:
            fill = False
            sub_sample = False
        
        # perform position-based sub-sampling if there are particles
        if sub_sample:
            # get x region for sub-sampling
            grid = data.Grid_Grid.data
            if x_select[0] is not None:
                x_min_sample = grid[0].min() + x_select[0]
            else:
                x_min_sample = grid[0].min()
            if x_select[1] is not None:
                x_max_sample = grid[0].min() + x_select[1]
            else:
                x_max_sample = grid[0].max()
            
            # check sub-sample region if valid
            if x_min_sample >= x_max_sample:
                raise ValueError("Invalid x_select")

            # perform sub-sample
            arr_inds = np.where(
                (x_temp<=x_max_sample) & (x_temp>=x_min_sample)
            )
            Ek_temp = Ek_temp[arr_inds]
            w_temp = w_temp[arr_inds]
            
            # check there are still particles
            if Ek_temp.size == 0:
                fill = False
        
        if fill:
            hist, _ = np.histogram(Ek_temp, bins=bin_edges, weights=w_temp)
            spectrum[:, i] = hist
            
        t[i] = data.Header['time']
    
    # check there is data to plot
    if spectrum.max() == 0:
        print("No electrons found")
        return
    
    if energy_units == 'gamma':
        bin_edges_plot = bin_edges / m_e / c ** 2 + 1
    elif energy_units == 'eV':
        bin_edges_plot = bin_edges / 1.609e-19
    else:
        raise ValueError('Invalid energy_units')
    
    # normalise
    spectrum *= e  # to charge
    
    # calculate some orders of magnitude    
    t_max_oom = int(np.floor(np.log10(t.max())))
    Ek_max_oom = int(np.floor(np.log10(bin_edges_plot.max())))
    spectrum_max_oom = int(np.floor(np.log10(spectrum.max())))
    
    # plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    im = ax.pcolormesh(
        t/10.0**t_max_oom,
        bin_edges_plot/10.0**Ek_max_oom,
        spectrum,
        shading='auto',
        cmap='hot_r'
    )
    cbar = fig.colorbar(im)
    ax.set_xlabel(r'$t$ ($\times$' + f'$10 ^ {{{t_max_oom}}}$' + ' s)')
    if energy_units == 'gamma':
        ax.set_ylabel(r'$\gamma$ ($\times$' + f'$10 ^ {{{Ek_max_oom}}}$' + ')')
    elif energy_units == 'eV':
        ax.set_ylabel(r'$E_k$ ($\times$' + f'$10 ^ {{{Ek_max_oom}}}$' + 
                      ' eV)')
    cbar.set_label(r'$q$ ($\times$' + f'$10 ^ {{{spectrum_max_oom}}}$' + ' C)')
    
    fig.tight_layout()
    
    plt.savefig(
        os.path.join(out_dir, 'electron_energy_spectrum.png'),
        dpi=dpi
    )
    
    plt.close()
    
    
    
def plot_back_of_the_bubble_velocity(
    sup_dir,
    n0=None,
    min_sep=None,
    t_start=None,
    t_end=None,
    plot_all=False,
    out_dir=None,
    figsize=(8, 4.5),
    dpi=200
):
    '''Plot the velocity of the back of the bubble.
    
    The back of the bubble is defined to be the point at which the longitudinal
    electric field has a root, in the first bucket behind the laser driver.
    
    The bubble velocity is undefined when a bubble doesn't exist, which can be
    the case before the laser has entered the plasma or if the laser energy has
    dropped significantly and no longer drives a wake.
    
    To find the temporal region in which the measurement is valid, first run
    this function without specifying `t_start` or `t_end` and setting `plot_all`
    to `True`. This will generate plots of the number density and electric field
    and indicate the best estimate for the position of the back of the bubble.
    Then run this function again with `plot_all` set to `False` and `t_start`
    and `t_end` set to encompass the time region over which the measurement is
    valid.
    
    Running with `plot_all=True` also allows for checking any erroneous
    measurements due to the minimum separation defined by `n0` or `min_sep`, and
    fine-tuning of this parameter.
    
    Parameters
    ----------
    sup_dir : str
        Path to the simulation super directory. The simulation dumps shouyld be
        in here.
    n0 : float, optional
        Background plasma density used to define the minimum separation between
        roots. Either this or `min_sep` must be specified. Defaults to None.
    min_sep : float, optional
        Minimum separation between roots in SI units. Either this or `n0` must
        be specified. Defaults to None.
    t_start : float, optional
        Lower boundary of time window within which to make measurements of the
        velocity. Defaults to None, which selects the earliest simulation time.
    t_end : float, optional
        Upper boundary of time window within which to make measurements of the
        velocity. Defaults to None, which selects the earliest simulation time.
    plot_all : bool, optional
        Whether or not to plot the number density, fields, roots and back of the
        bubble. If set to True, the back of the bubble velocity is not measured.
        Defaults to False.
    out_dir : str, optional
        Path to output directory within which to save the plot. Defaults to
        None, which saves to <sup_dir>/analysis/average_electron_energy/.
    figsize : tuple, optional
        Figure size in the form (width, height). Changing this parameter may
        result in a bad plot layout. Defaults to (8, 4.5).
    dpi : int, optional
        Dots per inch resolution. Changing this parameter may result in a bad
        plot layout. Defaults to 200.
        
    Input deck requirements
    -----------------------
    In addition to base requirements:
        - x-directional electric field.
        - Poynting flux.
        - Electron number density, if `plot_all` is `True`.
    '''
    # check one of n0 or min_sep has been provided
    if n0 is None and min_sep is None:
        raise ValueError("Either n0 or min_sep must be supplied.")
    if n0 is not None and min_sep is not None:
        raise ValueError("Only one of n0 or min_sep should be supplied.")
    
    # minimum separation between roots
    if n0 is not None:
        omega_p = e * np.sqrt(n0 / (epsilon_0 * m_e))  # plasma frequency
        lambda_p = 2 * np.pi * c / omega_p             # plasma wavelength
        min_sep = 0.2 * lambda_p
        
    if t_start is None:
        t_start = -1
    if t_end is None:
        t_end = np.infty
    
    # make output directory
    if out_dir is None:
        out_dir = os.path.join(sup_dir, "analysis", "back_of_bubble")
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
    
    # get time stamp of every dump
    t = np.zeros((len(sim_files, )), dtype=np.float64)
    for i, f in enumerate(sim_files):
        data = sdf.read(os.path.join(sup_dir, f))
        t[i] = data.Header['time']
        
    # dumps at times between t_start and t_end
    t_inds = np.where((t>t_start) & (t<t_end))
    if t_inds[0].size == 0:
        print("No dumps in time selection")
        return
    sim_files_select = np.array(sim_files)[t_inds]
    t_select = t[t_inds]

    # to be populated with back of the bubble positions
    bob_x = np.zeros((t_select.size, ), dtype=np.float64)
    
    # loop over simulation files
    for i, f in enumerate(tqdm(sim_files_select)):
        data = sdf.read(os.path.join(sup_dir, f))
        
        s = data.Electric_Field_Ex.data.T.shape[0]
        Ex = data.Electric_Field_Ex.data.T[s//2, :]
        Px = data.Derived_Poynting_Flux_x.data.T[s//2, :]
        grid = data.Grid_Grid_mid.data
        
        # spline and root find
        spline = CubicSpline(grid[0], Ex)
        roots_temp = spline.roots()
        # only deal with roots that are separated by more than the minimum
        roots = [roots_temp[0]]
        ind = 0
        for root in roots_temp[1:]:
            if root - roots[ind] > min_sep:
                roots.append(root)
                ind += 1
        roots = np.array(roots)
        
        # sub-sample roots with x-position less than the peak of the x-
        #   directional Poynting flux
        roots_select = roots[np.where(roots<grid[0][Px.argmax()])]
        
        # back of the bubble (bob) is typically the second-last root
        try:
            bob_x[i] = roots_select[-2]
        except IndexError:
            bob_x[i] = grid[0].min()
    
        # plot the fields and roots if requested
        if plot_all:
            n_e = data.Derived_Number_Density_electron.data.T

            fig, (ax0, ax1) = plt.subplots(2, figsize=figsize, dpi=dpi)
            ax0.imshow(
                n_e,
                origin='lower',
                aspect='auto',
                extent=(
                    grid[0][0],
                    grid[0][-1],
                    grid[1][0],
                    grid[1][-1]
                )
            )
            ax1.plot(grid[0], Ex, label='$E_x$')       
            ax1.plot(roots, np.zeros(roots.size), 'o', label='roots')
            ax1.plot(bob_x[i], 0, 'x', c='black', label='BoB')
            ax1.set_xlim(ax0.get_xlim())
            
            ax1.grid()
            ax0.set_xlabel('$x$ (m)')
            ax0.set_ylabel('$y$ (m)')
            ax1.set_xlabel('$x$ (m)')
            ax1.set_ylabel('$E_x$ (V/m)')
            fig.suptitle(f"$t = {data.Header['time']}$ s")
            fig.legend(loc='lower left')
            fig.tight_layout()
            
            fig.savefig(
                os.path.join(out_dir, f.split('.')[0] + '_bubble_position.png'),
                dpi=dpi
            )
            
            plt.close()
            
    if plot_all:
        return
    
    # assuming time-selection and roots are valid to calculate bubble velocity
    bob_v = np.diff(bob_x) / np.diff(t_select)
    
    # normalise to speed of light
    bob_v_c = bob_v / c
    
    # get mid-points of time
    t_plot = t_select[:-1] + np.diff(t_select) / 2
    
    # calculate some orders of magnitude
    t_max_oom = int(np.floor(np.log10(t_plot.max())))
    bob_v_max_plot = bob_v_c.max() * 1.1
    bob_v_max_plot_oom = int(np.floor(np.log10(bob_v_max_plot)))
    bob_v_max_plot /= 10.0 ** bob_v_max_plot_oom
    
    # plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    ax.plot(
        t_plot/10.0**t_max_oom,
        bob_v_c/10.0**bob_v_max_plot_oom
    )
    
    ax.set_xlim(t_plot.min()/10.0**t_max_oom, t_plot.max()/10.0**t_max_oom)
    ax.set_ylim(
        (bob_v_c.min()-(bob_v_max_plot-bob_v_c.max()))/10.0**bob_v_max_plot_oom,
        bob_v_max_plot
    )
    ax.set_xlabel(r'$t$ ($\times$' + f'$10 ^ {{{t_max_oom}}}$' + ' s)')
    ax.set_ylabel(r'$v_b$ ($\times$' + f'$10 ^ {{{bob_v_max_plot_oom}}}$' +
                  ' c)')
    ax.grid()
    
    fig.tight_layout()
    
    plt.savefig(
        os.path.join(out_dir, 'bubble_velocity.png'),
        dpi=dpi
    )
    
    plt.close()