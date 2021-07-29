Other diagnostics
=================

Back of the bubble position and velocity
----------------------------------------

The 'back of the bubble' (BoB) refers to the points behind the laser driver
where the longitudinal electric field is zero. It is of interest to electron
self-injection mechanics. See, for example, |text|_.

.. _text: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.044801
.. |text| replace:: Tooley *et al*. *Phys. Rev. Lett.* **119** (2017)

First, the interaction is plotted with the
:meth:`~visepoch.Simulation.plot_back_of_bubble_helper` method to visualise the
fields and determine the time window where the BoB measurements are valid.

::

   >>> sim.plot_back_of_bubble_helper(n0=1e25)

This generates several plots, some of which are shown below.

.. image:: /img/plot_back_of_bubble_helper_0005.png

.. image:: /img/plot_back_of_bubble_helper_0010.png

.. image:: /img/plot_back_of_bubble_helper_0020.png

The first image shows the laser entering the plasma. At this time, the bubble is
not fully formed, but is about to be.

The second image shows the interaction after some propagation
in the plasma. The bubble is fully formed.

The third image shows the laser leaving the plasma. The bubble is about to cease
to exist, so this is a good place to end the BoB measurements.

With this information, we can calculate the BoB position and velocity in the
time window those measurements are valid for.

::

   >>> sim.calculate_back_of_bubble_position_velocity(
   >>>     n0=1e25,
   >>>     t_start=3.4e-13,
   >>>     t_end=1.34e-12
   >>> )
   >>> sim.plot_back_of_bubble_velocity()

.. image:: /img/plot_back_of_bubble_velocity.png