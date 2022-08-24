import warnings
from pathlib import Path
import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm.notebook import tqdm as tqdm
from aeolus.const import init_const
from aeolus.plot import add_custom_legend, subplot_label_generator
from aeolus.synthobs import calc_stellar_flux, read_spectral_bands
from util_commons import PLANETS, SUITES

#Parameters
planet = "wasp96b"
metallicity = "solar"
exp = "equilibrium"

vrbls = {}
vrbls[planet] = {}
for exp in ["equilibrium", "kinetics"]:
    vrbls[planet][exp] = {}
    for metallicity in ["solar"]:
        # Load data
        star_constant_at_1_au = iris.cube.Cube(1272.8647540320192,long_name="stellar_constant_at_1_au",units="W m-2")
        path_to_spectral_file = SUITES[planet][exp][metallicity]["dir_for_star_spectrum_sw"]
        path_to_phase_curves = SUITES[planet][exp][metallicity][f"dir_for_phase_curves"]
        # SW EMISSION SPECTRUM
        sw_planet_flux = iris.load_cube(str(path_to_phase_curves), "m01s01i713")
        # LW EMISSION SPECTRUM
        lw_planet_flux = iris.load_cube(str(path_to_phase_curves), "m01s02i713")
        # Calculate stellar flux
        stellar_flux = calc_stellar_flux(path_to_spectral_file, star_constant_at_1_au)
        # Calculate planetary flux as a sum of the shortwave and longwave emission across the planet
        sw_planet_flux_sum = sw_planet_flux.collapsed(["latitude", "longitude"], iris.analysis.SUM)
        lw_planet_flux_sum = lw_planet_flux.collapsed(["latitude", "longitude"], iris.analysis.SUM)
        planet_flux = sw_planet_flux_sum + lw_planet_flux_sum
        planet_flux.coord("pseudo_level").rename("spectral_band_index")
        # Divide the planetary flux by the stellar flux
        fp_over_fs = planet_flux / stellar_flux
        # Find spectral band centers
        spectral_bands = read_spectral_bands(path_to_spectral_file)
        spectral_band_centers = 0.5 * (
            spectral_bands["lower_wavelength_limit"] + spectral_bands["upper_wavelength_limit"]
        )
        # Assemble data
        vrbls[planet][exp][metallicity] = {
            "spectral_band_centers": spectral_band_centers,
            "fp_over_fs": fp_over_fs,
        }
            
print(vrbls)