# WASP_96b
My scripts for WASP-96b Internship

calc_emission_spectrum - calculates synthetic emission spectrum for simulations (currently unfinished)

calc_transmission_spectrum - calculates synthetic transmission spectrum for simulations

check_burden - used to calculate the change in the amount of certian species through time

check_conservation - used to calculate the conservation of axial angular momentum and mass as well as the change in kinetic energy through time

check_eddy - used to calculate the zonal momentum budget using Denis's code apadted to my set up produces v similar plots to check_momentum

check_max_winds - used to plot the maximum value of each component of wind through time 

check_toa_net_energy_flux - used to check the balance of energy at the top of atmosphere

plt_gif - compiles all time slices for wind contours, species vertical profiles or PT structures into a single gif, also has code to make a set of directories

plt_mole_frac_on_plevs_along_latitudes - plots the vertical abundance profiles of CH4, NH3, H2O, CO, CO2 and HCN at all time slices

plt_temp_on_plevs_along_latitudes - plots the PT structure at every time slice

plt_winds - plots contours of the zonal meridonal and vertical mean for the zonal meridonal and vertical components of the wind at every time slice

plt_zonal_eddy_fluxes - plots figures a/b from Figure 7 Showman et al 2015 (currently uses height as vertical coordinate needs regridding to use pressure)

plt_zonal_momentum_budget - my method for plotting zonal momentum budget

plt_zonal_momentum_budget_Denis - Denis's method for plotting zonal momentum budget

proc_pp_to_nc - processes raw data into a single file with some new variable names erady for use in oter scripts

util_commons - dictionary containing paths to data for each simulation (need to remove initial PT and chemical profiles to make proc_pp_to_nc work)

HD209_commons - reduced dictionary used for testing of plt_zonal_momentum_budget on HD209 

