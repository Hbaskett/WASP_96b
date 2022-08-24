import warnings
from pathlib import Path

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm.notebook import tqdm as tqdm

from aeolus.const import init_const
from aeolus.coord import roll_cube_pm180
from aeolus.model import um
from aeolus.plot import add_custom_legend, subplot_label_generator
from aeolus.synthobs import (
    calc_stellar_flux,
    calc_transmission_spectrum,
    read_normalized_stellar_flux,
    read_spectral_bands,
)

from util_commons import GASES, PLANETS, SUITES
warnings.filterwarnings("ignore", module="iris")
warnings.filterwarnings("ignore", module="aeolus")

# Parameters
planet = "wasp96b"
exp = "equilibrium"
metallicity = "solar"
set = "JWST"
ty = "total"
sps = ["total", "NH3", "CH4"] #"co", "co2", "h2o", "hcn"]

def calc_transmission_spectrum_day_night_average(
    spectral_file,
    stellar_constant_at_1_au,
    stellar_radius,
    planet_top_of_atmosphere,
    planet_transmission_day,
    planet_transmission_night,
    model=um,
):

    # Ensure that input constants are iris cubes
    if not isinstance(stellar_constant_at_1_au, iris.cube.Cube):
        stellar_constant_at_1_au = iris.cube.Cube(
            stellar_constant_at_1_au,
            long_name="stellar_constant_at_1_au",
            units="W m-2",
        )
    if not isinstance(stellar_radius, iris.cube.Cube):
        stellar_radius = iris.cube.Cube(
            stellar_radius,
            long_name="stellar_radius",
            units="m",
        )
    if not isinstance(planet_top_of_atmosphere, iris.cube.Cube):
        planet_top_of_atmosphere = iris.cube.Cube(
            planet_top_of_atmosphere,
            long_name="planet_top_of_atmosphere",
            units="m",
        )

    # Load UM output from the dayside calculation
    day = planet_transmission_day
    day_lon_coord = day.coord(um.x)

    # Load UM output from the nightside calculation
    # Roll nightside data by 180 degrees
    night_rolled = roll_cube_pm180(planet_transmission_night)
    # Reverse longitude order (turned out to be not needed)
    # night = reverse(night_rolled, night_rolled.coord_dims(um.x))
    night = night_rolled
    # Replace the longitude coordinate to be able to do maths with iris
    night.replace_coord(day_lon_coord)

    # Use the same name for the spectral band index coordinate
    for coord in day.coords():
        if coord.name() in ["pseudo", "pseudo_level"]:
            coord.rename("spectral_band_index")
    for coord in night.coords():
        if coord.name() in ["pseudo", "pseudo_level"]:
            coord.rename("spectral_band_index")

    # Calculate the geometric mean of the dayside and nightside transmitted flux
    # and sum this flux over all latitudes and longitudes
    transmitted_flux = ((day * night) ** (0.5)).collapsed([um.y, um.x], iris.analysis.SUM)
    transmitted_flux.rename("total_transmitted_flux")
    transmitted_flux.units = "W m-2"

    # Calculate stellar flux
    stellar_flux = calc_stellar_flux(spectral_file, stellar_constant_at_1_au)

    # Calculate the ratio of the total transmitted flux to the stellar flux
    flux_ratio = transmitted_flux.copy(data=transmitted_flux.core_data() / stellar_flux.core_data())
    flux_ratio.rename("flux_ratio")
    flux_ratio.units = transmitted_flux.units / stellar_flux.units

    # Calculate the ratio of the effective planetary radius to the stellar radius
    rp_eff_over_rs_squared = (planet_top_of_atmosphere / stellar_radius) ** 2 - flux_ratio
    #rp_eff_over_rs_squared.data[rp_eff_over_rs_squared.data < 0.0] = 0.0
    rp_eff_over_rs = rp_eff_over_rs_squared ** (0.5)
    rp_eff_over_rs.rename("radius_ratio")

    # Find spectral band centers
    spectral_bands = read_spectral_bands(spectral_file)
    spectral_band_centers = 0.5 * (
        spectral_bands["lower_wavelength_limit"] + spectral_bands["upper_wavelength_limit"]
    )

    return spectral_band_centers, rp_eff_over_rs

stellar_constant_at_1_au = iris.cube.Cube(1272.8647540320192,long_name="stellar_constant_at_1_au",units="W m-2")
stellar_radius = iris.cube.Cube(730283400.0,long_name="stellar_radius",units="m")
planet_top_of_atmosphere = iris.cube.Cube(1.03e7 + 83893200.0,long_name="planet_top_of_atmosphere",units="m")

# Calculate UM transmission spectra (total and contributions from individual gases)
vrbls = {}
vrbls[planet] = {}
for exp in ["equilibrium", "kinetics"]:
    vrbls[planet][exp] = {}
    for metallicity in ["solar"]: #,"10xsolar"]:
        vrbls[planet][exp][metallicity] = {}
        for sp in tqdm(sps):
            in_dict = {
                "spectral_file": SUITES[planet][exp][metallicity]["dir_for_star_spectrum_sw"],
                "star_constant_at_1_au": stellar_constant_at_1_au,
                "star_radius": stellar_radius,
                "planet_top_of_atmosphere": planet_top_of_atmosphere,
                "planet_transmission_day": iris.load_cube(
                    str(SUITES[planet][exp][metallicity][f"dir_for_transmission_day_{sp}"])
                ),
                "planet_transmission_night": iris.load_cube(
                    str(SUITES[planet][exp][metallicity][f"dir_for_transmission_night_{sp}"])
                ),
            }
            spectral_band_centers, rp_eff_over_rs = calc_transmission_spectrum_day_night_average(
                *[value for key, value in in_dict.items()]
            )
            vrbls[planet][exp][metallicity][sp] = {
                "spectral_band_centers": spectral_band_centers,
                "rp_eff_over_rs": rp_eff_over_rs,
            }

# Import and process observational data
data_dir = Path.home()/ "um" / "um_runs" / "WASP_data"
JWST = np.loadtxt(data_dir / f"JWST_data.txt")
Nik = np.loadtxt(data_dir / f"Nikolov_2022_data.txt")

#Slice JWST data into useable arrays
JWST_wavelengths = JWST[:,0] #Extract wavelength data
Err_JWST_wavelengths = JWST[:,1] #Extract uncertainty in wavelength data
JWST_R_p_over_R_s_sqr = JWST[:,2] # Extract R_p over R_s 
Err_JWST_R_p_over_R_s_sqr = JWST[:,3] # extract uncertainty in R_p over R_s

# Slice Nikolov data
Nik_wavelengths = Nik[:,0]
Err_Nik_wavelengths = Nik[:,1]
Nik_R_p_over_R_s = Nik[:,2]
Err_Nik_R_p_over_R_s = Nik[:,3]

Nik_R_p_over_R_s_sqr = []
Err_Nik_R_p_over_R_s_sqr = []

# Need to square the data in the 3/4 th columns
for i in range(len(Nik_R_p_over_R_s)):
    a = Nik_R_p_over_R_s[i]**2
    Nik_R_p_over_R_s_sqr.append(a)

#Nikolov dataset gives standard deviation of datapoints this is the error from there plots, then multply by 2 and the value of R_p_over_R_s for error propagation
for i in range(len(Err_Nik_R_p_over_R_s)):
    a = 2*Nik_R_p_over_R_s[i]*Err_Nik_R_p_over_R_s[i]
    Err_Nik_R_p_over_R_s_sqr.append(a)
    
# Plotting dictionaries for total or single contributions
leg_line_kw_total = {"equilibrium": {"color": "orange","linestyle": "-"},
                      "kinetics": {"color": "green", "linestyle": "-"},
                      }

leg_line_kw_single = {"equilibrium": {"color": "orange","linestyle": "--"},
                      "kinetics": {"color": "green","linestyle": "--"},
                      }
#print(leg_line_kw_single.items(), "\n")
                      
#label = {
#    "equilibrium": {
#        "total": {"label": "total","color": "orange","linestyle": "-"},
#        "NH3": {"label": "NH3","color": "orange","linestyle": "--"},
#        "CH4": {"label": "CH4","color": "orange","linestyle": "--"},
#    },
#    "kinetics": {
#        "total": {"label": "total","color": "green","linestyle": "-"},
#        "NH3": {"label": "NH3","color": "green","linestyle": "--"},
#        "CH4": {"label": "CH4","color": "green","linestyle": "--"},
#    },
#}

#print(label.items(), "\n")
#print(label[exp].items(), "\n")

for metallicity in ["solar"]: #,"10xsolar"]:

    if metallicity == "solar":
        mdh = "MdH0"
    if metallicity == "10xsolar":
        mdh = "MdH1"
    
    for sp in tqdm(sps):
        # select plotting dictionary for total or single contributions
        #leg_line_kw = {}
        #if sp == "total":
        #    leg_line_kw = leg_line_kw_total
        #else:
        #    leg_line_kw = leg_line_kw_single
            
        # Add UM data
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6.75), constrained_layout=True, sharex=True)
        iletters = subplot_label_generator()
        for exp, plt_kw in .items():
            spectral_band_centers = vrbls[planet][exp][metallicity][sp]["spectral_band_centers"]
            rp_eff_over_rs = vrbls[planet][exp][metallicity][sp]["rp_eff_over_rs"]
            ax.plot(
                spectral_band_centers * 1e6,
                rp_eff_over_rs.data ** 2 - 0.0012,
                **plt_kw,
            )
        
        # Plot total transmission for comparison to single element contribution
        if sp != "total":
            for exp, ty, plt_kw in zip(["equilibrium","kinetics"],["total"], label[exp][ty].items()):
                spectral_band_centers = vrbls[planet][exp][metallicity]["total"]["spectral_band_centers"]
                rp_eff_over_rs = vrbls[planet][exp][metallicity]["total"]["rp_eff_over_rs"]
                ax.plot(
                    spectral_band_centers * 1e6,
                    rp_eff_over_rs.data ** 2 - 0.0012,
                    plt_kw,
                )
        
        # Only plot observational data for total transmission spectrum
        if sp == "total":
            # plot JWST errorbars
            plt.errorbar(
                JWST_wavelengths,
                JWST_R_p_over_R_s_sqr,
                xerr = Err_JWST_wavelengths,
                yerr = Err_JWST_R_p_over_R_s_sqr,
                label = "JWST",
                color = "red",
                fmt = 'o',
                alpha = 0.3,
            )
            # Plot Nikolov errorbars
            plt.errorbar(
                Nik_wavelengths,
                Nik_R_p_over_R_s_sqr,
                xerr = Err_Nik_wavelengths,
                yerr = Err_Nik_R_p_over_R_s_sqr,
                label = "Nikolov 2022",
                color = "blue",
                fmt = 'o',
                alpha = 0.3,
            )
    
        ax.set_xlim(0.2,12)
        
        # Set specifics for total contribution plot
        if sp == "total":  
            ax.set_ylim(0.013,0.0155)
            ax.set_xscale("log")
            ax.set_xlim(0.2,3)
            
        ax.set_title(f"{sp} Transmission Spectrum at {metallicity} Metallicity")
        ax.set_xlabel("Wavelength [$\mu$m]")
        ax.set_ylabel("($R_p/R_s)^2$")
        
        output_dir = Path.home() / "um" / "um_runs" / "trans"
        figname = f"{mdh}_trans_{sp}.png"
        fig.savefig(output_dir / figname)