import warnings
from pathlib import Path
import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from aeolus.coord import interp_cube_from_height_to_pressure_levels
from aeolus.model import um
from aeolus.plot import add_custom_legend, subplot_label_generator
from util_commons import PLANETS, SUITES, GASES

#Parameters
planet = "wasp96b"
exp = "equilibrium"
metallicity = "solar"
mdh = "MdH0"
output_dir = Path.home() / "um" / "um_runs" / "check_conservation"

vrbls = {}
vrbls[planet] = {}
for exp in ["equilibrium", "kinetics"]:
    vrbls[planet][exp] = {}
    for metallicity in ["solar","10xsolar"]:
        vrbls[planet][exp][metallicity] = {}
        fpath = SUITES[planet][exp][metallicity]["dir_for_raw"]
    
        df_eq1 = pd.DataFrame()
        # cannot pull from the merged files as they don't have the conservation info
        for data_file_eq1 in fpath.glob('**/atmos_base_*/conservation_diag.dat'):  # this directory and all subdirectories, recursively
    
        # f=open(data_file, "r")
        # lines = f.readlines()
        # for x in lines:
        #     flist.append(x.split()[0])
        # f.close()
          df_eq1 = pd.concat([df_eq1, pd.read_table(data_file_eq1, header=None, sep='   ', names=["timestep", "mass", "angular_momentum", "KE"], engine="python")])
    
        df_eq1.sort_values("timestep", inplace=True)
        print(df_eq1.head())
        df_eq1["mass"] = df_eq1["mass"].apply(lambda x: x / 2.410648e+23)
        df_eq1["angular_momentum"] = df_eq1["angular_momentum"].apply(lambda x: x / 2.446091e+34)
        df_eq1["KE"] = df_eq1["KE"].apply(lambda x: x / 9.950315e+23)
        df_eq1["timestep"] = df_eq1["timestep"].apply(lambda x: x/2880)
        # for label in zip(["mass", "angular_momentum", "KE"]):
        #     df[label] = df[label]

        # Assemble data
        vrbls[planet][exp][metallicity] = {
            "conserved": df_eq1,
        }

leg_line_kw = {"equilibrium": {"linestyle": "--"}, "kinetics": {"linestyle": "-"}}

for metallicity in ["solar","10xsolar"]:

    if metallicity == "solar":
        mdh = "MdH0"
    if metallicity == "10xsolar":
        mdh = "MdH1"    

    fig, axs = plt.subplots(1,3, figsize=(24,6))
    for exp, plt_kw in leg_line_kw.items():
        for label, ax in zip(["mass", "angular_momentum", "KE"], axs.reshape(-1)):
            ax.set_title(f"{label} conservation at {metallicity} metallicity")
            # Correctly title KE plot
            if label == "KE":
                ax.set_title(f"{label} change at {metallicity} metallicity")
            df = vrbls[planet][exp][metallicity]["conserved"]
            ax.plot(df["timestep"], df[label], linewidth=1, label = label)
            ax.set_ylabel(label)
            ax.set_xlabel("timestep / days")
        
    figname = f"{mdh}_energy.png"
    fig.savefig(output_dir / figname)
    
