import warnings
from pathlib import Path

import iris
import iris.pandas
import matplotlib.pyplot as plt
import numpy as np

from aeolus.calc.diag import toa_eff_temp
from aeolus.calc.stats import spatial_mean
from aeolus.coord import get_cube_rel_days, interp_to_cube_time
from aeolus.model import um
from aeolus.plot import add_custom_legend, subplot_label_generator
from util_commons import GASES, SUITES, PLANETS
warnings.filterwarnings("ignore", module="iris")

planet = "wasp96b"
exp = "equilibrium"
metallicity = "solar"
mdh = "MdH0"
output_dir = Path.home() / "um" / "um_runs" / "check_TOA"
figname = f"{mdh}_TOA.png"


# Process data
vrbls = {}
for planet in PLANETS.keys():
    vrbls[planet] = {}
    for exp in ["equilibrium", "kinetics"]:
        vrbls[planet][exp] = {}
        for metallicity in ["solar","10xsolar"]:
            # Load data
            fpath = SUITES[planet][exp][metallicity]["dir_for_merged"]
            fname = f"{SUITES[planet][exp][metallicity]['suite']}.nc"
            toa_isr = iris.load_cube(str(fpath / fname), um.toa_isr)  # toa_incoming_shortwave_flux
            _toa_olr = iris.load_cube(str(fpath / fname), um.toa_olr)  # toa_outgoing_longwave_flux
            toa_osr = iris.load_cube(str(fpath / fname), um.toa_osr)  # toa_outgoing_shortwave_flux
            # Remove forecast_period coordinate
            toa_isr.remove_coord("forecast_period")
            _toa_olr.remove_coord("forecast_period")
            toa_osr.remove_coord("forecast_period")
            # Interpolate _toa_olr to toa_isr's time coordinate
            toa_olr = interp_to_cube_time(_toa_olr, toa_isr, model=um)
            # Calculate global mean TOA incoming, outgoing and net energy fluxes
            toa_isr_mean = spatial_mean(toa_isr, model=um)
            toa_olr_mean = spatial_mean(toa_olr, model=um)
            toa_net_mean = spatial_mean(toa_isr - toa_olr - toa_osr, model=um)
            toa_isr_mean.rename("global_mean_toa_incoming_energy_flux")
            toa_net_mean.rename("global_mean_toa_net_energy_flux")
            # Find number of simulation days since the first model output on day 10
            time_in_rel_days = get_cube_rel_days(toa_isr, model=um) + 10
            # Assemble data
            vrbls[planet][exp][metallicity] = {
                "time_in_rel_days": time_in_rel_days,
                "toa_isr_mean": toa_isr_mean,
                "toa_olr_mean": toa_olr_mean,
                "toa_net_mean": toa_net_mean,
            }

# Print final values the global mean TOA incoming energy flux
#for planet in PLANETS.keys():
#    for exp in ["equilibrium", "kinetics"]:
#        print(planet, exp, vrbls[planet][exp]["toa_isr_mean"][-1].data / 1e3)  # [kW m-2]

# Print final values for the global mean TOA net energy flux
#for planet in PLANETS.keys():
#    for exp in ["equilibrium", "kinetics"]:
#        print(planet, exp, vrbls[planet][exp]["toa_net_mean"][-1].data / 1e3)  # [kW m-2]

# Print final values for the relationship between global mean incoming and net energy fluxes
# net as a % of incoming
#for planet in PLANETS.keys():
#    for exp in ["equilibrium", "kinetics"]:
#        print(
#            planet,
#            exp,
#            vrbls[planet][exp]["toa_net_mean"][-1].data
#            * 100
#            / vrbls[planet][exp]["toa_isr_mean"][-1].data,
#        )  # [%]

# Calculate effective planetary temperature
#for planet in PLANETS.keys():
#    toa_temp_eff_by_planet = []
#    for exp in ["equilibrium", "kinetics"]:   
#        cl = iris.cube.CubeList()
#        cl.append(vrbls[planet][exp][metallicity]["toa_olr_mean"][-1])
#        toa_temp_eff_by_planet.append(toa_eff_temp(cl).data)
#             print(planet, exp, toa_eff_temp(cl).data)  # K
#    print(planet, round(np.array(toa_temp_eff_by_planet).mean()))  # K
    

# Plot TOA net energy flux vs time
for metallicity in ["solar","10xsolar"]:

    if metallicity == "solar":
        mdh = "MdH0"
    if metallicity == "10xsolar":
        mdh = "MdH1"

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 6.75), sharex=True, constrained_layout=True)
    iletters = subplot_label_generator()
    for exp, linestyle in zip(["equilibrium", "kinetics"], ["dashed", "solid"]):
        ax.plot(
            vrbls[planet][exp][metallicity]["time_in_rel_days"],
            vrbls[planet][exp][metallicity]["toa_net_mean"].data,
            linestyle=linestyle,
            color="k",
        )
    
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Global mean TOA net energy flux [W $m^{-2}$]")
        if ax.get_subplotspec().is_first_row():
            add_custom_legend(
                ax,
                {
                    "equilibrium".capitalize(): {"linestyle": "dashed", "color": "k"},
                    "kinetics".capitalize(): {"linestyle": "solid", "color": "k"},
                },
                frameon=False,
                loc="upper right",
        )
    
    ax.set_ylabel("Global mean TOA net energy flux [W $m^{-2}$]")
    ax.set_xlim(0, 1000)
    ax.set_title(f"{metallicity} Metallicity top of atmosphere flux")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time [days]")
    output_dir = Path.home() / "um" / "um_runs" / "check_TOA"
    figname = f"{mdh}_TOA.png"
    fig.savefig(output_dir / figname)


for metallicity in ["solar","10xsolar"]:

    if metallicity == "solar":
        mdh = "MdH0"
    if metallicity == "10xsolar":
        mdh = "MdH1"

    # Plot % change in TOA net energy fluxes starting on day 10 vs time
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 6.75), sharex=True, constrained_layout=True)
    iletters = subplot_label_generator()
    for exp, linestyle in zip(["equilibrium", "kinetics"], ["dashed", "solid"]):
        ax.plot(
            vrbls[planet][exp][metallicity]["time_in_rel_days"],
            iris.pandas.as_series(vrbls[planet][exp][metallicity]["toa_net_mean"], copy=True)
            .pct_change()
            .fillna(0)
            .add(1)
            .cumprod()
            .sub(1),
            linestyle=linestyle,
            color="k",
        )
    
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("% of global mean TOA net energy flux on day 10")
        if ax.get_subplotspec().is_first_row():
            add_custom_legend(
                ax,
                {
                    "equilibrium".capitalize(): {"linestyle": "dashed", "color": "k"},
                    "kinetics".capitalize(): {"linestyle": "solid", "color": "k"},
                },
                frameon=False,
                loc="upper right",
        
            )

    #iax.set_xlim(0, 1000)
    #iax.set_ylim(bottom=0)
    ax.set_title(f"{metallicity} Metallicity Percentage change in top of atmosphere flux")
    ax.set_ylabel("% of global mean TOA net energy flux on day 10")
    ax.set_xlabel("Time [days]")
    output_dir = Path.home() / "um" / "um_runs" / "check_TOA_percent"
    figname = f"{mdh}_Percentage_TOA.png"
    fig.savefig(output_dir / figname)
