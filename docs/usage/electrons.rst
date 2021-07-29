Electron diagnostics
====================

Electron beam energy
--------------------

:: 

   >>> _ = sim.calculate_electron_energy_avg_std()
   >>> sim.plot_electron_energy_avg_std()

.. image:: /img/plot_electron_energy_avg_std.png


Electron beam charge
--------------------

::

   >>> _ = sim.calculate_electron_charge()
   >>> sim.plot_electron_charge()

.. image:: /img/plot_electron_charge.png

Electron energy spectrum
------------------------

::

   >>> _ = sim.calculate_electron_energy_spectrum(energy_bins=100, subset='high_gamma')
   >>> sim.plot_electron_energy_spectrum()

.. image:: /img/plot_electron_energy_spectrum.png