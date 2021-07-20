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
from scipy.constants import e

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
    ax.set_ylabel(r'$E$ ($\times$' + f'$10 ^ {{{Ek_max_plot_oom}}}$' + ' eV)')
    
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
    
    
    
