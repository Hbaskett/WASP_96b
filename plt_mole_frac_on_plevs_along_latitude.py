import warnings
from pathlib import Path
import iris
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from aeolus.coord import interp_cube_from_height_to_pressure_levels
from aeolus.model import um
from aeolus.plot import add_custom_legend, subplot_label_generator
from util_commons import PLANETS, SUITES, GASES
from tqdm.notebook import tqdm as tqdm

warnings.filterwarnings("ignore", module="iris")
warnings.filterwarnings("ignore", module="aeolus")

#When plotting new species change
# species variable
# plot title + x axis label
# name in _species
# mdh variable

# Parameters
metallicity = "solar"
mdh = "MdH0"
gases = ["CH4", "CO", "CO2", "H2O", "HCN", "NH3"]
t = 99 # 80th slice of 10-day-frequency output, i.e. 800 day
day = (t+1)*10 # day of simulation
tgt_plevs = np.logspace(7, 2, 65)  # target pressure levels
tgt_lats = [0, 60]  # target latitudes
tgt_lons = np.arange(0, 360, 30) + 1.25  # target longitudes*
interp_args = ([("latitude", tgt_lats), ("longitude", tgt_lons)], iris.analysis.Linear())
output_dir = Path.home() / "um" / "um_runs" / "chem_lats"

# *in the UM v11.4 substellar point in radiation is shifted 1.25 degrees east

vrbls = {}
for planet in PLANETS.keys():
    vrbls[planet] = {}
    for exp in ["equilibrium", "kinetics"]:
        vrbls[planet][exp] = {}
        for metallicity in ["solar","10xsolar"]:
            vrbls[planet][exp][metallicity] = {}
            
            if metallicity == "solar":
                t=99
                day = (t*10)+10
            if metallicity == "10xsolar":
                t=69
                day = (t*10)+10
            
            for gas in tqdm(gases):
                vrbls[planet][exp][metallicity][gas] = {}
                # Find other data
                fpath = SUITES[planet][exp][metallicity]["dir_for_merged"]
                fname = f"{SUITES[planet][exp][metallicity]['suite']}.nc"
                # Load variables
                pres = iris.load_cube(str(fpath / fname), um.pres)[-1, ...]
                _species = iris.load_cube(str(fpath/fname), f"{gas} mole fraction")[t:, ...]
                species = _species.collapsed("time", iris.analysis.MEAN) # need to average across all time steps to match the size of the plotted cubes
                # Interpolate Mole fraction to pressure levels
                species_plevs = interp_cube_from_height_to_pressure_levels(species, pres, tgt_plevs)
                species_plevs_locs = species_plevs.interpolate(*interp_args)
                pres_coord = species_plevs_locs.coord("air_pressure")
                # Assemble data
                vrbls[planet][exp][metallicity][gas] = {
                    "species_plevs_locs": species_plevs_locs,
                }

# Plotting parameters
# Sample each colormap 4 times but use the brightest 3 colours from each colormap
nclrs = 4
clrs_1 = plt.cm.Blues_r(np.linspace(0, 1, nclrs)[0 : nclrs - 1])
clrs_2 = plt.cm.Oranges(np.linspace(0, 1, nclrs)[1:])
clrs_3 = plt.cm.Reds_r(np.linspace(0, 1, nclrs)[0 : nclrs - 1])
clrs_4 = plt.cm.Greens(np.linspace(0, 1, nclrs)[1:])
lon_clrs = np.concatenate([clrs_1, clrs_2, clrs_3, clrs_4])

alpha_60 = 0.4
lon_clrs_60 = lon_clrs * np.array([1, 1, 1, alpha_60])
leg_lat_kw = {
    "0": {"linestyle": "-", "linewidth": 2, "color": "grey"},
    "60°N": {"linestyle": "-", "linewidth": 1, "color": "grey", "alpha": alpha_60},
}

leg_exp_kw = {
    "Equilibrium": {"linestyle": "--", "linewidth": 2, "color": "k"},
    "Kinetics": {"linestyle": "-", "linewidth": 2, "color": "k"},
}
leg_line_kw = {"equilibrium": {"linestyle": "--"}, "kinetics": {"linestyle": "-"}}
leg_init_kw = {"Initial": {"linestyle": (0, (1, 1)), "linewidth": 2, "color": "k"}}

# Plot
for metallicity in ["solar","10xsolar"]:

    if metallicity == "solar":
        mdh = "MdH0"
        t=99
        day = (t*10)+10
    if metallicity == "10xsolar":
        mdh = "MdH1"
        t=69
        day = (t*10)+10
        
    for gas in tqdm(gases):
    
        fig, ax = plt.subplots(
            ncols=1, nrows=1, figsize=(10, 6.75), sharex=True, sharey=True, constrained_layout=True,
        )
        iletters = subplot_label_generator()
        for exp, plt_kw in leg_line_kw.items():
            for lon, clr, clr60 in zip(tgt_lons, lon_clrs, lon_clrs_60):
                vp00 = vrbls[planet][exp][metallicity][gas]["species_plevs_locs"].extract(
                    iris.Constraint(latitude=0, longitude=lon)
                )
                vp60 = vrbls[planet][exp][metallicity][gas]["species_plevs_locs"].extract(
                    iris.Constraint(latitude=60, longitude=lon)
                )
                ax.plot(
                    vp00.data,
                    pres_coord.points,
                    **plt_kw,
                    color=clr,
                    label=f"{int(lon-1.25)}°",
                    linewidth=2,
                )
                ax.plot(
                    vp60.data,
                    pres_coord.points,
                    **plt_kw,
                    color=clr60,
                    label=f"{int(lon-1.25)}°",
                    linewidth=1,
                )
    
        # Add extra legends
        add_custom_legend(
            ax, leg_exp_kw, loc="center left", bbox_to_anchor=(0, 0.22), frameon=False, fontsize="small"
        )
        add_custom_legend(
            ax, leg_lat_kw, loc="center left", bbox_to_anchor=(0, 0.12), frameon=False, fontsize="small"
        )
        add_custom_legend(
            ax, leg_init_kw, loc="lower left", bbox_to_anchor=(0, -0.005), frameon=False, fontsize="small"
        )
        
        # Add longitudes legend
        # Delete repeating handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()
        i = 1
        while i < len(labels):
            if labels[i] in labels[:i]:
                del labels[i]
                del handles[i]
            else:
                i += 1
                
        leg = ax.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(-0.04, 1),
            frameon=False,
            handlelength=0,
            fontsize="small",
        )
        
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
    
        ax.set_title(f"Mole fraction of {gas} at day {day}")
        ax.set_xlabel(f"{gas} Mole Fraction")
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Pressure [Pa]")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(1e7, 1e2)
        ax.grid()    
        figname = f"{mdh}_{gas}_Chem_{day}.png"
        fig.savefig(output_dir / figname)
