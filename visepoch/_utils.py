# Author: George K. Holt
# License: MIT
# Version: 0.2.0
"""Part of VISEPOCH.

Contains utility functions for data manipulation and file handling.
"""
import numpy as np
import os
import matplotlib.pyplot as plt



def find_nearest_idx(a, v):
    '''Return index of numpy array a with value closest to v.'''
    a = np.asarray(a)
    return (np.abs(a - v)).argmin()



def weighted_avg_std(arr, w):
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



def calculate_oom(v):
    '''Calculate order of magnitude of real numberical value v.'''
    return int(np.floor(np.log10(v)))



def create_outdir(sup_dir, dirname='visepoch'):
    '''Create output directory under sim_dir if it doesn't exist.'''
    if not os.path.isdir(os.path.join(sup_dir, dirname)):
        os.makedirs(os.path.join(sup_dir, dirname))
    return os.path.join(sup_dir, dirname)



def save_or_show(fig, savefig, fname, out_dir=None, sup_dir=None):
    '''Save of show a figure.
    
    Parameters
    ----------
    fig : matplotlib figure instance
    savefig : bool
        True to save, False to show.
    out_dir : str
        Path to output directory. Set to None to auto create a subdirectory of
        sup_dir.
    sup_dir : str
        Path to parent directory within which to create output directory. Only
        required if out_dir is None.
    '''    
    if savefig:
        if out_dir is None:
            out_dir = create_outdir(sup_dir)
        else:
            if not os.path.isdir(out_dir):
                raise ValueError("out_dir does not exist")
        fig.savefig(os.path.join(out_dir, fname))
        plt.close(fig)
    else:
        plt.show()