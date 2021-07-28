# Author: George K. Holt
# License: MIT
# Version: 0.2.0
"""Part of VISEPOCH.

Contains functions for handling EPOCH simulation data.
"""
import os
import numpy as np
import sdf



def datafiles(self, directory):
    '''Returns an ordered list of EPOCH simulation files.'''
    sim_files = []
    for file in os.listdir(directory):
        if file.endswith('.sdf'):
            sim_files.append(file)
    sim_files.sort()
    return sim_files



def timestamps(self, directory, sim_files):
    '''Returns 1D numpy array of simulation timestamps.'''
    t = np.zeros((len(sim_files),), dtype=np.float64)
    for i, f in enumerate(sim_files):
        data = sdf.read(os.path.join(directory, f))
        t[i] = data.Header['time']
    return t



def x_sub_sample(grid, x_select):
    '''Sub-sample particle positions based on x-coordinate.
    
    Parameters
    ----------
    grid : numpy ndarray
        1D array of x-position data.
    x_select : tuple
        2-tuple of of float values specifying x-position of particles to sub-
        sample in SI units (xmin, xmax) relative to minimum boundary. Value of
        None sets it to minimum or maximum border.
        
    Returns
    -------
    arr_inds : tuple
        1-tuple containing numpy ndarray of array indices of `grid` that lie in
        the region defined by `x_select`.
    '''
    # get x-region
    if x_select[0] is not None:
        x_min_sample = grid.min() + x_select[0]
    else:
        x_min_sample = grid.min()
    if x_select[1] is not None:
        x_max_sample = grid.min() + x_select[1]
    else:
        x_max_sample = grid.max()
    
    # check valid
    if x_min_sample >= x_max_sample:
        raise ValueError("Invalid x_select")

    # get indices of sub-sample
    arr_inds = np.where(
        (grid<=x_max_sample) & (grid>=x_min_sample)
    )
    
    return arr_inds