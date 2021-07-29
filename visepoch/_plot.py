# Author: George K. Holt
# License: MIT
# Version: 0.2.0
"""
_plot.py
========

Part of VISEPOCH.

Contains functions for plotting.  These should not be accessed directly as they
are imported by the Simulation class.
"""
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.constants import c, m_e, e

import sdf

from ._utils import calculate_oom, create_outdir, save_or_show



################################################################################
################################################################################
# *** ELECTRON PLOTS
################################################################################
################################################################################

def get_mpl_style(mpl_style):
    """Return path to matplotlib style sheet."""
    if mpl_style == 'default':
        mpl_stylesheet_path = os.path.join(
            os.path.split(__file__)[0],
            '_mpl_config',
            'style.mplstyle'
        )
    else:
        mpl_stylesheet_path = mpl_style
        # check custom sheet exists
        if not os.path.isfile(mpl_style):
            raise ValueError("Custom mpl_style does not exist")
    
    return mpl_stylesheet_path



def plot_electron_energy_avg_std(
    self,
    savefig=False,
    out_dir=None,
    mpl_style='default',
    energy_units='eV',
):
    """Create plot of average electron energy against time.
    
    Parameters
    ----------
    savefig : bool, optional
        Whether to save the plot (``True``) or show it (``False``). Defaults to
        ``False``.
    out_dir : str, optional
        Output directory within which to save the plot. Only used if `savefig`
        is ``True``. Defaults to ``None`` which saves to a directory called
        'visepoch' created under the simulation directory.
    mpl_style : str, optional
        Either ``'default'``, which uses the visepoch matplotlib style, or a
        path to a matplotlib style sheet. Using a non-default style will likely
        break the plot. Defaults to ``'default'``.
    energy_units : {'eV', 'J'}
        Units to plot energy in. One of ``['eV', 'J']``. Defaults to
        ``'eV'``.
    """
    # check if data is available
    if self.electron_energy_avg_std is None:
        raise RuntimeError("No average and standard deviation electron energy "
                        "data")
        
    mpl_stylesheet_path = get_mpl_style(mpl_style)
    
    # convert energy units
    if energy_units == 'eV':
        Ek_avg_plot = self.electron_energy_avg_std[0] / 1.609e-19
        Ek_std_plot = self.electron_energy_avg_std[1] / 1.609e-19
    elif energy_units == 'J':
        Ek_avg_plot = self.electron_energy_avg_std[0]
        Ek_std_plot = self.electron_energy_avg_std[1]
    else:
        raise ValueError("Invalid energy_units")
    
    # calculate some axis limits and orders of magnitude
    t_max_oom = calculate_oom(self.t.max())
    Ek_max_plot = np.add(Ek_avg_plot, Ek_std_plot).max() * 1.1
    Ek_max_plot_oom = calculate_oom(Ek_max_plot)
    Ek_max_plot /= 10.0 ** Ek_max_plot_oom
    
    # plot
    with plt.style.context(mpl_stylesheet_path):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.fill_between(
            self.t/10.0**t_max_oom,
            Ek_avg_plot/10.0**Ek_max_plot_oom-Ek_std_plot/10.0**Ek_max_plot_oom,
            Ek_avg_plot/10.0**Ek_max_plot_oom+Ek_std_plot/10.0**Ek_max_plot_oom,
            alpha=0.3,
            color='C1'
        )
        ax.plot(
            self.t/10.0**t_max_oom,
            Ek_avg_plot/10.0**Ek_max_plot_oom,
            c='C1'
        )
        ax.set_xlim(
            self.t.min()/10.0**t_max_oom,
            self.t.max()/10.0**t_max_oom
        )
        ax.set_ylim(0, Ek_max_plot)
        ax.set_xlabel(r'$t$ ($\times$' + f'$10 ^ {{{t_max_oom}}}$' + ' s)')
        if energy_units == 'eV':
            ax.set_ylabel(r'$E_k$ ($\times$' + f'$10 ^ {{{Ek_max_plot_oom}}}$' +
                          ' eV)')
        elif energy_units == 'J':
            ax.set_ylabel(r'$E_k$ ($\times$' + f'$10 ^ {{{Ek_max_plot_oom}}}$' +
                          ' J)')
        
        fig.tight_layout()
        
        save_or_show(
            fig,
            savefig,
            'electron_energy_avg_std.png',
            out_dir,
            self.directory
        )
            
            
            
def plot_electron_charge(
    self,
    savefig=False,
    out_dir=None,
    mpl_style='default'
):
    """Create plot of electron charge against time.
    
    Parameters
    ----------
    savefig : bool, optional
        Whether to save the plot (``True``) or show it (``False``). Defaults to
        ``False``.
    out_dir : str, optional
        Output directory within which to save the plot. Only used if `savefig`
        is ``True``. Defaults to ``None`` which saves to a directory called
        'visepoch' created under the simulation directory.
    mpl_style : str, optional
        Either ``'default'``, which uses the visepoch matplotlib style, or a
        path to a matplotlib style sheet. Using a non-default style will likely
        break the plot. Defaults to ``'default'``.
    """
    # check if data is available
    if self.electron_charge is None:
        raise RuntimeError("No electron charge data")
    
    mpl_stylesheet_path = get_mpl_style(mpl_style)

    # calculate some axis limits and orders of magnitude
    t_max_oom = calculate_oom(self.t.max())
    q_max_plot = self.electron_charge.max() * 1.1
    q_max_plot_oom = calculate_oom(q_max_plot)
    q_max_plot /= 10.0 ** q_max_plot_oom
    
    # plot
    with plt.style.context(mpl_stylesheet_path):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(
            self.t/10.0**t_max_oom,
            self.electron_charge/10.0**q_max_plot_oom,
            c='C1'
        )
        ax.set_xlim(self.t.min()/10.0**t_max_oom, self.t.max()/10.0**t_max_oom)
        ax.set_ylim(0, q_max_plot)
        ax.set_xlabel(r'$t$ ($\times$' + f'$10 ^ {{{t_max_oom}}}$' + ' s)')
        ax.set_ylabel(r'$q$ ($\times$' + f'$10 ^ {{{q_max_plot_oom}}}$' + ' C)')
        
        fig.tight_layout()
        
        save_or_show(
            fig,
            savefig,
            'electron_charge.png',
            out_dir,
            self.directory
        )
            
            
            
def plot_electron_energy_spectrum(
    self,
    energy_units='gamma',
    savefig=False,
    out_dir=None,
    mpl_style='default'
):
    """Create plot of electron energy spectrum against time.
    
    Parameters
    ----------
    energy_units : {'gamma', 'eV'}
        Units to plot the energy axis in. One of ``('gamma', 'eV')``. Defaults
        to ``'gamma'``.
    savefig : bool, optional
        Whether to save the plot (``True``) or show it (``False``). Defaults to
        ``False``.
    out_dir : str, optional
        Output directory within which to save the plot. Only used if `savefig`
        is ``True``. Defaults to ``None`` which saves to a directory called
        'visepoch' created under the simulation directory.
    mpl_style : str, optional
        Either ``'default'``, which uses the visepoch matplotlib style, or a
        path to a matplotlib style sheet. Using a non-default style will likely
        break the plot. Defaults to ``'default'``.
    """
    # check if data is available
    if self.electron_energy_spectrum is None:
        raise RuntimeError("No electron spectrum data")
    
    mpl_stylesheet_path = get_mpl_style(mpl_style)
    
    if energy_units == 'gamma':
        bin_edges_plot = self.electron_energy_spectrum[0] / m_e / c ** 2 + 1
    elif energy_units == 'eV':
        bin_edges_plot = self.electron_energy_spectrum[0] / 1.609e-19
    else:
        raise ValueError("Invalid energy units")
        
    # normalise
    spectrum = self.electron_energy_spectrum[1] * e  # to charge
    
    # calculate some orders of magnitude
    t_max_oom = calculate_oom(self.t.max())
    Ek_max_oom = calculate_oom(bin_edges_plot.max())
    spectrum_max_oom = calculate_oom(spectrum.max())
    
    # plot
    with plt.style.context(mpl_stylesheet_path):
        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.pcolormesh(
            self.t/10.0**t_max_oom,
            bin_edges_plot/10.0**Ek_max_oom,
            spectrum/10.0**spectrum_max_oom,
            shading='auto',
            cmap='hot_r'
        )
        cbar = fig.colorbar(im)
        ax.set_xlabel(r'$t$ ($\times$' + f'$10 ^ {{{t_max_oom}}}$' + ' s)')
        if energy_units == 'gamma':
            ax.set_ylabel(r'$\gamma$ ($\times$' + f'$10 ^ {{{Ek_max_oom}}}$' +
                          ')')
        elif energy_units == 'eV':
            ax.set_ylabel(r'$E_k$ ($\times$' + f'$10 ^ {{{Ek_max_oom}}}$' + 
                          ' eV)')
        cbar.set_label(r'$q$ ($\times$' + f'$10 ^ {{{spectrum_max_oom}}}$' +
                       ' C)')
        
        fig.tight_layout()
        
        save_or_show(
            fig,
            savefig,
            'electron_energy_spectrum.png',
            out_dir,
            self.directory
        )
            
            
            
################################################################################
################################################################################
# *** LASER PLOTS
################################################################################
################################################################################

def plot_laser_a0(
    self,
    savefig=False,
    out_dir=None,
    mpl_style='default'
):
    """Create a plot of laser dimensionless amplitude against time.
    
    Parameters
    ----------
    savefig : bool, optional
        Whether to save the plot (``True``) or show it (``False``). Defaults to
        ``False``.
    out_dir : str, optional
        Output directory within which to save the plot. Only used if `savefig`
        is ``True``. Defaults to ``None`` which saves to a directory called
        'visepoch' created under the simulation directory.
    mpl_style : str, optional
        Either ``'default'``, which uses the visepoch matplotlib style, or a
        path to a matplotlib style sheet. Using a non-default style will likely
        break the plot. Defaults to ``'default'``.
    """
    # check if data is available
    if self.laser_a0 is None:
        raise RuntimeError("No laser a0 data")
    
    mpl_stylesheet_path = get_mpl_style(mpl_style)
    
    # calculate some axis limits and orders of magnitude
    t_max_oom = calculate_oom(self.t.max())
    a0_max_plot = self.laser_a0.max() * 1.1
    a0_max_plot_oom = calculate_oom(self.laser_a0.max())
    a0_max_plot /= 10.0 ** a0_max_plot_oom
    
    # plot
    with plt.style.context(mpl_stylesheet_path):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(
            self.t/10.0**t_max_oom,
            self.laser_a0/10.0**a0_max_plot_oom,
            c='C0'
        )
        ax.set_xlim(self.t.min()/10.0**t_max_oom, self.t.max()/10.0**t_max_oom)
        ax.set_ylim(0, a0_max_plot)
        ax.set_xlabel(r'$t$ ($\times$' + f'$10 ^ {{{t_max_oom}}}$' + ' s)')
        ax.set_ylabel(r'$a_0$ ($\times$' + f'$10 ^ {{{a0_max_plot_oom}}}$' +
                      ')')
        
        fig.tight_layout()
        
        save_or_show(
            fig,
            savefig,
            'laser_a0.png',
            out_dir,
            self.directory
        )
        
        
        
################################################################################
################################################################################
# *** OTHER PLOTS
################################################################################
################################################################################

def plot_back_of_bubble_helper(
    self,
    n0=None,
    min_sep=None,
    out_dir=None,
    mpl_style='default',
):
    """Create plots of the plasma number density and calculated back of the
    bubble position for every dump.
    
    This is useful for determining the valid time range for
    :meth:`visepoch.simulation.Simulation.calculate_back_of_bubble_position_velocity`.
    
    Parameters
    ----------
    n0 : float, optional
        Background plasma density used to define the minimum separation between
        roots. Either this or `min_sep` must be specified. Defaults to ``None``.
    min_sep : float, optional
        Minimum separation between roots in SI units. Either this or `n0` must
        be specified. Defaults to ``None``.
    out_dir : str, optional
        Output directory within which to save the plot. Only used if `savefig`
        is ``True``. Defaults to ``None`` which saves to a directory called
        'visepoch' created under the simulation directory.
    mpl_style : str, optional
        Either ``'default'``, which uses the visepoch matplotlib style, or a
        path to a matplotlib style sheet. Using a non-default style will likely
        break the plot. Defaults to ``'default'``.
    
    Notes
    -----
    
    | Input deck requirements:
    | - ``Ex = always``
    | - ``poynting_flux = always``
    | - ``number_density = always + species``
    """
    bob_x_temp, bob_v_temp, t_mid_temp = \
        self.calculate_back_of_bubble_position_velocity(
            n0=n0, min_sep=min_sep, force=True, set_var=False
        )
    
    mpl_stylesheet_path = get_mpl_style(mpl_style)
    
    with plt.style.context(mpl_stylesheet_path):
        for i, f in enumerate(self.sim_files):
            data = sdf.read(os.path.join(self.directory, f))
            n_e = data.Derived_Number_Density_electron.data.T
            s = data.Electric_Field_Ex.data.T.shape[0]
            Ex = data.Electric_Field_Ex.data.T[s//2, :]
            grid = data.Grid_Grid_mid.data
            
            
            fig, (ax0, ax1) = plt.subplots(2)
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
            ax0.grid(False)
            ax1.plot(grid[0], Ex, label='$E_x$')
            ax1.plot(bob_x_temp[i], 0, 'x', c='black', label='BoB')
            ax1.set_xlim(ax0.get_xlim())
            
            ax0.set_xlabel('$x$ (m)')
            ax0.set_ylabel('$y$ (m)')
            ax1.set_xlabel('$x$ (m)')
            ax1.set_ylabel('$E_x$ (V/m)')
            fig.suptitle(f"$t = {data.Header['time']}$ s")
            fig.legend(loc='lower left')
            fig.tight_layout()
            
            if out_dir is None:
                create_outdir(self.directory)
                out_dir = create_outdir(
                    os.path.join(self.directory, 'visepoch'),
                    'back_of_bubble_helper'
                )
            else:
                if not os.path.isdir(out_dir):
                    raise ValueError("out_dir does not exist")
            
            fig.savefig(os.path.join(
                out_dir,
                f.split('.')[0] + '_bubble_position.png'
            ))
            
            plt.close(fig)
            
            
            
def plot_back_of_bubble_velocity(
    self,
    savefig=False,
    out_dir=None,
    mpl_style='default'
):
    """Create plot of back of the bubble velocity against time.
    
    Parameters
    ----------
    savefig : bool, optional
        Whether to save the plot (``True``) or show it (``False``). Defaults to
        ``False``.
    out_dir : str, optional
        Output directory within which to save the plot. Only used if `savefig`
        is ``True``. Defaults to ``None`` which saves to a directory called
        'visepoch' created under the simulation directory.
    mpl_style : str, optional
        Either ``'default'``, which uses the visepoch matplotlib style, or a
        path to a matplotlib style sheet. Using a non-default style will likely
        break the plot. Defaults to ``'default'``.
    """
    # check if data is available
    if self.back_of_bubble_velocity is None:
        raise RuntimeError("No back of the bubble data")
    
    mpl_stylesheet_path = get_mpl_style(mpl_style)
    
    back_of_bubble_v_c = self.back_of_bubble_velocity / c
    
    # calculate some orders of magnitude
    t_max_oom = calculate_oom(self.back_of_bubble_t_mid.max())
    
    # plot
    with plt.style.context(mpl_stylesheet_path):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(
            self.back_of_bubble_t_mid/10.0**t_max_oom,
            back_of_bubble_v_c
        )
        
        ax.set_xlim(
            self.back_of_bubble_t_mid.min()/10.0**t_max_oom,
            self.back_of_bubble_t_mid.max()/10.0**t_max_oom
        )
        ax.set_ylim(
            back_of_bubble_v_c.min() - 0.1 * back_of_bubble_v_c.max(),
            back_of_bubble_v_c.max() * 1.1
        )
        
        ax.set_xlabel(r'$t$ ($\times$' + f'$10 ^ {{{t_max_oom}}}$' + ' s)')
        ax.set_ylabel(r'$v_b$ (c)')
        
        fig.tight_layout()
        
        save_or_show(
            fig,
            savefig,
            'back_of_bubble_velocity.png',
            out_dir,
            self.directory
        )