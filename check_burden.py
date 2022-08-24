import warnings
from pathlib import Path
import iris
import iris.pandas
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm as tqdm

from aeolus.const import init_const
from aeolus.coord import (
    get_cube_rel_days,
    interp_cube_from_height_to_pressure_levels,
    volume_weights_cube,
)
from aeolus.model import um
from aeolus.plot import add_custom_legend, subplot_label_generator
from util_commons import GASES, SUITES, PLANETS
warnings.filterwarnings("ignore", module="iris")

consts = init_const()
universal_gas_const = consts.molar_gas_constant
planet = "wasp96b"
exp = "equilibrium"
metallicity = "solar"
mdh = "MdH0"
gases = ["CH4", "CO", "CO2", "H2O", "HCN", "NH3"]
burden_unit = "Eg"
log_pres_thresholds = [3, 4, 5, 6, None]
output_dir = Path.home() / "um" / "um_runs" / "check_burden"

# Process data
vrbls = {}
vrbls[planet] = {}
for exp in ["equilibrium", "kinetics"]:
  vrbls[planet][exp] = {}
  for metallicity in ["solar","10xsolar"]:
      vrbls[planet][exp][metallicity] = {}
      for gas in tqdm(gases):
          vrbls[planet][exp][metallicity][gas] = {}
          for log_pres_threshold in log_pres_thresholds:
              if log_pres_threshold is not None:
                  thresh = 10 ** log_pres_threshold
              else:
                  thresh = 1e99  # practically nothing is masked
              # Load data
              fpath = SUITES[planet][exp][metallicity]["dir_for_merged"]
              fname = f"{SUITES[planet][exp][metallicity]['suite']}.nc"
              air_temperature = iris.load_cube(str(fpath / fname), um.temp)
              air_pressure = iris.load_cube(str(fpath / fname), um.pres)
              mole_frac = iris.load_cube(str(fpath / fname), f"{gas} mole fraction")
              # Calculate grid cell volumes
              volume = volume_weights_cube(
                  air_temperature, r_planet=83893200.0, model=um
              )
              # Calculate mass of the gas in each grid cell
              mass = (
                      air_pressure
                      * volume
                      * mole_frac
                      * GASES[gas]["molar_mass"]
                      / (air_temperature * universal_gas_const)
              )
              # Mask mass data where air pressure is higher than a selected threshold
              mass = mass.copy(data=np.ma.masked_where(air_pressure.data > thresh, mass.data))
              # Calculate the total mass of the gas (aka burden)
              burden = mass.collapsed(
                  ["level_height", "latitude", "longitude"], iris.analysis.SUM
              )
              #burden.convert_units(burden_unit)
              # Number of simulation days since the first model output on day 10
              time_in_rel_days = get_cube_rel_days(mole_frac, model=um) + 10
              # Assemble data
              vrbls[planet][exp][metallicity][gas][str(log_pres_threshold)] = {
                  "time_in_rel_days": time_in_rel_days,
                  "burden": burden,
              }

# Print final values of the % change of the total initial burden; select more than 1% changes

for gas in gases:
    for log_pres_threshold in [str(i) for i in log_pres_thresholds]:
        y = (
            iris.pandas.as_series(
                vrbls[planet][exp][metallicity][gas][log_pres_threshold]["burden"], copy=True
            )
            .pct_change()
            .fillna(0)
            .add(1)
            .cumprod()
            .sub(1)[-1]
        )
        if y < -1 or y > 1:
            print(gas, log_pres_threshold, y)


# Choose layers to plot
plt_log_pres_thresholds = [4, 5]

# Plot
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
        for gas in gases:
            clr = GASES[gas.upper()]["color"]
            for log_pres_threshold, alpha in zip(
                [str(i) for i in plt_log_pres_thresholds], [1, 0.5]
            ):
                ax.plot(
                    vrbls[planet][exp][metallicity][gas][log_pres_threshold]["time_in_rel_days"],
                    iris.pandas.as_series(
                        vrbls[planet][exp][metallicity][gas][log_pres_threshold]["burden"], copy=True
                    )
                    .pct_change()
                    .fillna(0)
                    .add(1)
                    .cumprod()
                    .sub(1),
                    linestyle=linestyle,
                    color=clr,
                    alpha=alpha,
                )
                
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel(f"% of the initial total mass")
        if ax.get_subplotspec().is_first_row():
            add_custom_legend(
                ax,
                {
                    "$CH_4$": {"color": "C3"},
                    "$CO$": {"color": "C1"},
                    "$CO_2$": {"color": "C2"},
                    "$H_2O$": {"color": "C0"},
                    "$HCN$": {"color": "C4"},
                    "$NH_3$": {"color": "C5"},
                },
                frameon=False,
                loc="upper left",
            )
            add_custom_legend(
                ax,
                {
                    "$10^{2}$-$10^{4}$ Pa": {"color": "grey", "alpha": 1},
                    "$10^{2}$-$10^{5}$ Pa": {"color": "grey", "alpha": 0.5},
                    #"$10^{2}$-$10^{3}$ Pa": {"color": "grey", "alpha": 1},
                    #"$10^{2}$-$10^{4}$ Pa": {"color": "grey", "alpha": 0.65},
                    #"$10^{2}$-$10^{5}$ Pa": {"color": "grey", "alpha": 0.40},
                    #"$10^{2}$-$10^{6}$ Pa": {"color": "grey", "alpha": 0.20},
                    #"$10^{2}$-$10^{7}$ Pa": {"color": "grey", "alpha": 0.10},
                },
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

    ax.set_xlim(0, 1000)
    ax.set_ylim(-1, 5)
    ax.set_title(f"{metallicity} Metallicity Burden")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(f"% of the initial total mass")
    figname = f"{mdh}_Burden.png"
    fig.savefig(output_dir / figname)