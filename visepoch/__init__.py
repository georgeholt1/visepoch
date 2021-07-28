class Simulation:
    
    from ._data import (
        datafiles,
        timestamps
    )
    
    from ._analyse import (
        calculate_electron_energy_avg_std,
        calculate_electron_charge,
        calculate_electron_energy_spectrum,
        calculate_laser_a0,
        calculate_back_of_bubble_position_velocity
    )
    from ._plot import (
        plot_electron_energy_avg_std,
        plot_electron_charge,
        plot_electron_energy_spectrum,
        plot_laser_a0,
        plot_back_of_bubble_helper,
        plot_back_of_bubble_velocity
    )
    
    def __init__(self, directory):
        
        self.directory = directory
        self.sim_files = self.datafiles(self.directory)
        self.t = self.timestamps(self.directory, self.sim_files)
        
        # electron beam measurements
        self.electron_energy_avg_std = None
        self.electron_charge = None
        self.electron_energy_spectrum = None
        
        # laser beam measurements
        self.laser_a0 = None
        
        # other measurements
        self.back_of_bubble_position = None
        self.back_of_bubble_velocity = None
        self.back_of_bubble_t_mid = None