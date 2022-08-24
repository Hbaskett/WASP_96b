import warnings
from pathlib import Path

import iris
import iris.pandas
import matplotlib.pyplot as plt

from aeolus.coord import get_cube_rel_days
from aeolus.model import um
from aeolus.plot import add_custom_legend, subplot_label_generator
from util_commons import GASES, PLANETS, SUITES

warnings.filterwarnings("ignore", module="iris")
planet = "wasp96b"
exp = "equilibrium"
metallicity = "solar"
mdh = "MdH0"
output_dir = Path.home() / "um" / "um_runs" / "check_winds"
figname = f"{mdh}_Winds.png"


# Process data
vrbls = {}
vrbls[planet] = {}
for metallicity in ["solar","10xsolar"]:
    vrbls[planet][metallicity] = {}
    for exp in ["equilibrium", "kinetics"]:
        # Load data
        fpath = SUITES[planet][exp][metallicity]["dir_for_merged"]
        fname = f"{SUITES[planet][exp][metallicity]['suite']}.nc"
        u = iris.load_cube(str(fpath / fname), um.u)
        v = iris.load_cube(str(fpath / fname), um.v)
        w = iris.load_cube(str(fpath / fname), um.w) * 100  # scaled for plotting
        # Calculate max zonal, meridional and vertical wind velocity
        u_max = u.collapsed(["level_height", "latitude", "longitude"], iris.analysis.MAX)
        v_max = v.collapsed(["level_height", "latitude", "longitude"], iris.analysis.MAX)
        w_max = w.collapsed(["level_height", "latitude", "longitude"], iris.analysis.MAX)
        # Find number of simulation days since the first model output on day 10
        time_in_rel_days = get_cube_rel_days(u, model=um) + 10
        # Assemble data
        vrbls[planet][metallicity][exp] = {
            "time_in_rel_days": time_in_rel_days,
            "u_max": u_max,
            "v_max": v_max,
            "w_max": w_max,
        }

# Print final values
#for planet in PLANETS.keys():
#    for exp in ["equilibrium", "kinetics"]:
#        print(
#            planet,
#            exp,
#            vrbls[planet][exp]["u_max"][-1].data / 1e3,
#            vrbls[planet][exp]["v_max"][-1].data / 1e3,
#            vrbls[planet][exp]["w_max"][-1].data / 1e5,
#        )

# Plot wind velocities vs time
for metallicity in ["solar","10xsolar"]:

    if metallicity == "solar":
        mdh = "MdH0"
    if metallicity == "10xsolar":
        mdh = "MdH1"

    fig, ax = plt.subplots(
        ncols=1, nrows=1, figsize=(12, 6.75), sharex=True, sharey=True, constrained_layout=True
    )
    iletters = subplot_label_generator()
    for exp, linestyle in zip(["equilibrium", "kinetics"], ["dashed", "solid"]):
        ax.plot(
            vrbls[planet][metallicity][exp]["time_in_rel_days"],
            vrbls[planet][metallicity][exp]["u_max"].data,
            color="b",
            linestyle=linestyle,
        )
        ax.plot(
            vrbls[planet][metallicity][exp]["time_in_rel_days"],
            vrbls[planet][metallicity][exp]["v_max"].data,
            color="g",
            linestyle=linestyle,
        )
        ax.plot(
            vrbls[planet][metallicity][exp]["time_in_rel_days"],
            vrbls[planet][metallicity][exp]["w_max"].data,
            color="r",
            linestyle=linestyle,
        )
    
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Wind velocity [m$s^{-1}$]")
        if ax.get_subplotspec().is_first_row():
            add_custom_legend(
                ax,
                {"u": {"color": "b"}, "v": {"color": "g"}, r"w($\times100$)": {"color": "r"}},
                frameon=False,
                loc="upper center",
            )
            add_custom_legend(
                ax,
                {
                    "equilibrium".capitalize(): {"linestyle": "dashed", "color": "k"},
                    "kinetics".capitalize(): {"linestyle": "solid", "color": "k"},
                },
                frameon=False,
                loc="upper right",
            )
        
    ax.set_ylabel("Wind velocity [m$s^{-1}$]")
    ax.set_xlabel("Time [days]")
    ax.grid()
    ax.set_title(f"{metallicity} Metallicity Average Wind Velocities")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 7000)
    output_dir = Path.home() / "um" / "um_runs" / "check_winds"
    figname = f"{mdh}_Winds.png"
    fig.savefig(output_dir / figname)


# Plot % change in wind velocities starting on day 10 vs time
for metallicity in ["solar","10xsolar"]:

    if metallicity == "solar":
        mdh = "MdH0"
    if metallicity == "10xsolar":
        mdh = "MdH1"

    fig, ax = plt.subplots(
        ncols=1, nrows=1, figsize=(12, 6.75), sharex=True, sharey=True, constrained_layout=True
    )
    iletters = subplot_label_generator()
    for exp, linestyle in zip(["equilibrium", "kinetics"], ["dashed", "solid"]):
        ax.plot(
            vrbls[planet][metallicity][exp]["time_in_rel_days"],
            iris.pandas.as_series(vrbls[planet][metallicity][exp]["u_max"], copy=True)
            .pct_change()
            .fillna(0)
            .add(1)
            .cumprod()
            .sub(1),
            color="b",
            linestyle=linestyle,
        )
        ax.plot(
            vrbls[planet][metallicity][exp]["time_in_rel_days"],
            iris.pandas.as_series(vrbls[planet][metallicity][exp]["v_max"], copy=True)
            .pct_change()
            .fillna(0)
            .add(1)
            .cumprod()
            .sub(1),
            color="g",
            linestyle=linestyle,
        )
        ax.plot(
            vrbls[planet][metallicity][exp]["time_in_rel_days"],
            iris.pandas.as_series(vrbls[planet][metallicity][exp]["w_max"], copy=True)
            .pct_change()
            .fillna(0)
            .add(1)
            .cumprod()
            .sub(1),
            color="r",
            linestyle=linestyle,
        )

    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("% of wind velocity on day 10")
        if ax.get_subplotspec().is_first_row():
            add_custom_legend(
                ax,
                {"u": {"color": "b"}, "v": {"color": "g"}, r"w($\times100$)": {"color": "r"}},
                frameon=False,
                loc="upper left",
            )
            add_custom_legend(
            ax,
                {
                    "equilibrium".capitalize(): {"linestyle": "dashed", "color": "k"},
                    "kinetics".capitalize(): {"linestyle": "solid", "color": "k"},
                },
                frameon=False,
                loc="center right",
            )
        
    ax.set_ylabel("% of wind velocity on day 10")
    ax.set_xlabel("Time [days]")
    ax.set_title(f"{metallicity} Metallicity Percentage Change in Average Wind Velocities")
    ax.set_xlim(0, 1000)
    output_dir = Path.home() / "um" / "um_runs" / "check_winds_percent"
    figname = f"{mdh}_Percentage_Winds.png"
    fig.savefig(output_dir / figname)
