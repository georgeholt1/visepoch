# Author: George K. Holt
# License: MIT
# Version: 0.2.0
"""Part of VISEPOCH.

Contains functions for analysing EPOCH simulation data.
"""
import numpy as np
import os
from scipy.constants import e, epsilon_0, m_e, c
from scipy.interpolate import CubicSpline

import sdf

from ._data import x_sub_sample
from ._utils import weighted_avg_std



################################################################################
################################################################################
# *** ELECTRON MEASUREMENTS
################################################################################
################################################################################

def calculate_electron_energy_avg_std(
    self,
    x_select=(None, None),
    force=False,
    set_var=True
):
    '''Calculate average and standard deviation of energy of electrons in
    subset.
    
    Parameters
    ----------
    x_select : tuple, optional
        2-tuple with float values specifying the x-position of particles to
        sub-sample in SI relative window units (xmin, xmax). A value of None
        sets the sub-sample to be the minimum of maximum x-border.
        E.g. (5e-6, 20e-6) will sample the particles with position values
        between 5 and 20 microns in the window units.
        Defaults to (None, None), which performs no sub-sampling.
    force : bool, optional
        Set to True to force the get, even if data has already been got. Useful
        if `x_select` changes. Defaults to False.
    set_var : bool, optional
        Whether to set (and possibly overwrite) the class variable. Defaults to
        True.
    
    Input deck requirements
    -----------------------
    - `high_gamma` subset of `electron` species.
    - Particle `position`, `weight` and `energy` dumped for `high_gamma` subset.
    
    Returns
    -------
    Ek_avg : numpy ndarray
        1D array of average energy values.
    Ek_std : numpy ndarray
        1D array of weighted standard deviation values.
    '''
    # check if this data is already available
    if self.electron_energy_avg_std is not None:
        if not force:
            return self.electron_energy_avg_std
    
    # initialise arrays
    Ek_avg = np.zeros((len(self.sim_files), ), dtype=np.float64)
    Ek_std = np.zeros((len(self.sim_files), ), dtype=np.float64)
    
    # loop over data files
    for i, f in enumerate(self.sim_files):
        data = sdf.read(os.path.join(self.directory, f))
        # if AttributeError, there are no particles in subset
        try:
            Ek_temp = data.Particles_Ek_subset_high_gamma_electron.data
            w_temp = data.Particles_Weight_subset_high_gamma_electron.data
            x_temp = data.Grid_Particles_subset_high_gamma_electron.data[0]
            sub_sample = True
        except AttributeError:
            Ek_temp = 0
            w_temp = 1
            sub_sample = False
        
        if sub_sample:
            arr_inds = x_sub_sample(x_temp,  x_select)
            Ek_temp = Ek_temp[arr_inds]
            w_temp = w_temp[arr_inds]
            
            # check there are still particles
            if Ek_temp.size == 0:
                Ek_temp = 0
                w_temp = 1
        
        Ek_avg_temp, Ek_std_temp = weighted_avg_std(Ek_temp, w_temp)
        
        Ek_avg[i] = Ek_avg_temp
        Ek_std[i] = Ek_std_temp
        
    if set_var:
        self.electron_energy_avg_std = (Ek_avg, Ek_std)

    return Ek_avg, Ek_std



def calculate_electron_charge(
    self,
    x_select=(None, None),
    force=False,
    set_var=True
):
    '''Calculate total charge of electrons in subset.
    
    Parameters
    ----------
    x_select : tuple, optional
        2-tuple with float values specifying the x-position of particles to
        sub-sample in SI relative window units (xmin, xmax). A value of None
        sets the sub-sample to be the minimum of maximum x-border.
        E.g. (5e-6, 20e-6) will sample the particles with position values
        between 5 and 20 microns in the window units.
        Defaults to (None, None), which performs no sub-sampling.
    force : bool, optional
        Set to True to force the get, even if data has already been got. Useful
        if `x_select` changes. Defaults to False.
    set_var : bool, optional
        Whether to set (and possibly overwrite) the class variable. Defaults to
        True.
        
    Input deck requirements
    -----------------------
    - `high_gamma` subset of `electron` species.
    - Particle `position` and `weight` dumped for `high_gamma` subset.
    
    Returns
    -------
    q : numpy ndarray
        1D array of electron charge.
    '''
    # check if data is already available
    if self.electron_charge is not None:
        if not force:
            return self.electron_charge
    
    # initialise array
    q = np.zeros((len(self.sim_files), ), dtype=np.float64)
    
    # loop over data files
    for i, f in enumerate(self.sim_files):
        data = sdf.read(os.path.join(self.directory, f))
        # if AttributeError, there are no particles in subset
        try:
            w_temp = data.Particles_Weight_subset_high_gamma_electron.data
            x_temp = data.Grid_Particles_subset_high_gamma_electron.data[0]
            sub_sample = True
        except AttributeError:
            w_temp = 0
            sub_sample = False
            
        if sub_sample:
            arr_inds = x_sub_sample(x_temp, x_select)
            w_temp = w_temp[arr_inds]
            
            # check there are still particles
            if w_temp.size == 0:
                w_temp = 0
        
        # convert sum of weights to charge
        q_temp = np.sum(w_temp) * e
        q[i] = q_temp
    
    if set_var:
        self.electron_charge = q
    
    return q



def calculate_electron_energy_spectrum(
    self,
    energy_bins=1000,
    subset='all',
    x_select=(None, None),
    force=False,
    set_var=True
):
    '''Calculate electron energy spectrum.
    
    Parameters
    ----------
    energy_bins : int, optional
        Number of bins to form the spectrum in energy. Defaults to 1000.
    subset : str, optional
        The subset of electron to calculate the spectrum for. Can be 'all' or
        'high_gamma'. Defaults to 'all'.
    x_select : tuple, optional
        2-tuple with float values specifying the x-position of particles to
        sub-sample in SI relative window units (xmin, xmax). A value of None
        sets the sub-sample to be the minimum of maximum x-border.
        E.g. (5e-6, 20e-6) will sample the particles with position values
        between 5 and 20 microns in the window units.
        Defaults to (None, None), which performs no sub-sampling.
    force : bool, optional
        Set to True to force the get, even if data has already been got. Useful
        if `x_select` changes. Defaults to False.
    set_var : bool, optional
        Whether to set (and possibly overwrite) the class variable. Defaults to
        True.
        
    Input deck requirements
    -----------------------
    - If subset is `high_gamma`, then a subset of the `electron` species called
      `high_gamma`.
    - Particle `position`, `weight` and `energy` dumped for the `electron`
      species if subset is `all`, or for the `high_gamma` subset if `subset` is
      `high_gamma`.
      
    Returns
    -------
    bin_edges : numpy ndarray
        1D array of histogram bin edges.
    spectrum : numpy ndarray
        2D array with energy spectrum with energy in axis 0 and time in axis 1.
    '''
    # check subset valid
    if subset not in ('all', 'high_gamma'):
        raise ValueError("Invalid subset")
    
    # check if data is already available
    if self.electron_energy_spectrum is not None:
        if not force:
            return self.electron_energy_spectrum
        
    # initialise array
    spectrum = np.zeros(
        (energy_bins, len(self.sim_files)),
        dtype=np.float64
    )
    
    def _get_particles(data, subset):
        '''Get particle data.
        
        Parameters
        ----------
        data : sdf instance
        subset : str
            Either 'all' or 'high_gamma'.
        
        Returns
        -------
        3-tuple of 1D numpy ndarrays with (energy, weight, x-position data) or
        None if no particles found.
        '''
        if subset == 'high_gamma':
            try:
                return (
                    data.Particles_Ek_subset_high_gamma_electron.data,
                    data.Particles_Weight_subset_high_gamma_electron.data,
                    data.Grid_Particles_subset_high_gamma_electron.data[0]
                )
            except AttributeError:
                return None
        elif subset == 'all':
            try:
                return (
                    data.Particles_Ek_electron.data,
                    data.Particles_Weight_electron.data,
                    data.Grid_Particles_electron.data[0]
                )
            except AttributeError:
                return None
    
    # get maximum energy in simulation
    Ek_max = 0
    for f in self.sim_files:
        data = sdf.read(os.path.join(self.directory, f))
        parts_temp = _get_particles(data, subset)
        if parts_temp is not None and parts_temp[0].max() > Ek_max:
            Ek_max = parts_temp[0].max()
    
    # histogram bin edges
    bin_edges = np.linspace(0, Ek_max, energy_bins+1)
    
    # calculate histogram and populate the energy spectrum image-like
    for i, f in enumerate(self.sim_files):
        data = sdf.read(os.path.join(self.directory, f))
        parts_temp = _get_particles(data, subset)
        if parts_temp is not None:
            sub_sample = True
            Ek_temp, w_temp, x_temp = parts_temp
            fill = True
        else:
            sub_sample = False
            fill = False
        
        # perform position-based sub-sampling if there are particles
        if sub_sample:
            arr_inds = x_sub_sample(x_temp, x_select)
            w_temp = w_temp[arr_inds]
            Ek_temp = Ek_temp[arr_inds]
            
            # check there are still particles
            if Ek_temp.size == 0:
                fill = False
                
        if fill:
            hist, _ = np.histogram(Ek_temp, bins=bin_edges, weights=w_temp)
            spectrum[:, i] = hist
    
    if set_var:
        self.electron_energy_spectrum = (bin_edges, spectrum)

    return (bin_edges, spectrum)
            
    
    
    
################################################################################
################################################################################
# *** LASER MEASUREMENTS
################################################################################
################################################################################

def calculate_laser_a0(
    self,
    cpw_x,
    lambda0=800e-9,
    force=False,
    set_var=True
):
    '''Calculate peak on-axis laser dimensionless amplitude.
    
    Parameters
    ----------
    cpw_x : int
        Cells per laser wavelength in the x-direction.
    lambda0 : float, optional
        Laser central wavelength in SI units. Defaults to 800e-9.
    force : bool, optional
        Set to True to force the get, even if data has already been got. Useful
        if `x_select` changes. Defaults to False.
    set_var : bool, optional
        Whether to set (and possibly overwrite) the class variable. Defaults to
        True.
        
    Input deck requirements
    -----------------------
    - poynting_flux
    
    Returns
    -------
    a0 : numpy ndarray
        1D array of laser dimensionless amplitude values.
    '''
    # check if data is already available
    if self.laser_a0 is not None:
        if not force:
            return self.laser_a0
    
    # initialise array
    a0 = np.zeros((len(self.sim_files), ), dtype=np.float64)
    
    # loop over data files
    for i, f in enumerate(self.sim_files):
        data = sdf.read(os.path.join(self.directory, f))
        
        # lineout of x-directional Poynting flux
        s = data.Derived_Poynting_Flux_x.data.T.shape[0]
        Px = data.Derived_Poynting_Flux_x.data.T[s//2, :]
        
        # moving average with window size equal to cells per wavelength
        Px_ma = np.convolve(Px, np.ones(cpw_x), 'valid') / cpw_x
        
        # peak intensity
        I0 = Px_ma.max()
        
        # a0 = 0.855 * lanbda0 [micron] * sqrt(I [10^18 W/cm^2])
        a0[i] = 0.855 * lambda0 * 1e6 * np.sqrt(I0 * 1e-22)
        
    if set_var:
        self.laser_a0 = a0

    return a0



################################################################################
################################################################################
# *** OTHER MEASUREMENTS
################################################################################
################################################################################

def calculate_back_of_bubble_position_velocity(
    self,
    n0=None,
    min_sep=None,
    t_start=None,
    t_end=None,
    force=False,
    set_var=True
):
    '''Calculate back of the bubble position and velocity.
    
    The back of the bubble is defined to be the point at which the longitudinal
    electric field has a root, in the first bucket behind the laser driver.
    
    The bubble velocity is undefined when a bubble doesn't exist, which can be
    the case before the laser has entered the plasma or if the laser energy has
    dropped significantly and no longer drives a wake.
    
    Use the plotting helper function `plot_back_of_bubble_helper` to find the
    time region over which the measurements are valid.
    
    Parameters
    ----------
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
    force : bool, optional
        Set to True to force the get, even if data has already been got. Useful
        if `x_select` changes. Defaults to False.
    set_var : bool, optional
        Whether to set (and possibly overwrite) the class variable. Defaults to
        True.
    
    Input deck requirements
    -----------------------
    - x-directional electric field.
    - Poynting flux.
    
    Returns
    -------
    bob_x : numpy ndarray
        1D array of x-position of back of the bubble values.
    bob_v : numpy ndarray
        1D array of velocity of back of the bubble values.
    t_mid : numpy ndarray
        1D array of time values corresponding to `bob_v`.
    '''
    # check if data already available
    if self.back_of_bubble_velocity is not None:
        if not force:
            return self.back_of_bubble
        
    # check one of n0 or min_sep provided
    if n0 is None and min_sep is None:
        raise ValueError("Either n0 or min_sep must be supplied")
    if n0 is not None and min_sep is not None:
        raise ValueError("Only one of n0 or min_sep should be supplied")
    
    # minimum separation between roots
    if n0 is not None:
        omega_p = e * np.sqrt(n0 / (epsilon_0 * m_e))  # plasma frequency
        lambda_p = 2 * np.pi * c / omega_p             # plasma wavelength
        min_sep = 0.2 * lambda_p
        
    if t_start is None:
        t_start = -1
    if t_end is None:
        t_end = np.infty

    # dumps at times between t_start and t_end
    t_inds = np.where((self.t>t_start) & (self.t<t_end))
    if t_inds[0].size == 0:
        raise RuntimeError("No dumps in this time range.")
    sim_files_select = np.array(self.sim_files)[t_inds]
    t_select = self.t[t_inds]
    
    # to be populated with back of the bubble positions
    bob_x = np.zeros((t_select.size, ), dtype=np.float64)
    
    # loop over simulation files
    for i, f in enumerate(sim_files_select):
        data = sdf.read(os.path.join(self.directory, f))
        
        # load on-axis data
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
            
    
    # calculate bubble velocity
    bob_v = np.diff(bob_x) / np.diff(t_select)
    
    # mid-points of time
    t_mid = t_select[:-1] + np.diff(t_select) / 2
    
    if set_var:
        self.back_of_bubble_position = bob_x
        self.back_of_bubble_velocity = bob_v
        self.back_of_bubble_t_mid = t_mid
        
    return (bob_x, bob_v, t_mid)