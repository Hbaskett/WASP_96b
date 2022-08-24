import warnings
from pathlib import Path
import numpy as np
import scipy as sp
import iris
import iris.pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
from iris.analysis.maths import apply_ufunc, divide
from aeolus.calc import deriv
from aeolus.coord import interp_cube_from_height_to_pressure_levels, regrid_3d, coord_to_cube
from aeolus.model import um
from aeolus.plot import add_custom_legend, subplot_label_generator
#from util_commons import GASES, PLANETS, SUITES
from HD209_commons import SUITES

# Functions but not giving correct scale outputs ideas are:
# - np.gradient wrong?
# - maths need full radius but then how to plot in the range?

#Parameters
warnings.filterwarnings("ignore", module="iris")
planet = "hd209458b"
exp = "equilibrium"
metallicity = "solar"
mdh = "MdH0"
spc_mdh0 = 3516.2947232694723
spc_mdh1 = 3164.69353854304
spc_HD209 =  3516.6
comp = "meridonal"
moment = "momentum"
limit = 1 # Use 0 to plot zoomed data set at jet or full data set
t = 0 # time slice integer
day = (t*10)+10
tgt_plevs = np.logspace(7, 2, 65)  # target pressure levels
output_dir = Path.home() / "um" / "um_runs" / "check_conservation" 

# Denis's function for calculating the derivative with latitude
def d_dphi(cube, const=None, model=um):
    r"""
    Calculate a derivative w.r.t. latitude.
    .. math::
        \frac{1}{r}\frac{\partal A}{\partial \phi}
    """
    ycoord = cube.coord(model.y).copy()
    cube.coord(model.y).convert_units("radians")
    out = deriv(cube, model.y)
    out /= 0.9e7 #1.03e7 # divide out by the radius
    cube.coord(model.y).convert_units("degrees")
    cube.replace_coord(ycoord)
    out.coord(model.y).convert_units("degrees")
    out.replace_coord(ycoord)
    return out

# Process data
vrbls = {}
vrbls[planet] = {}
for exp in ["equilibrium", "kinetics"]:
    vrbls[planet][exp] = {}
    for metallicity in ["solar"]: #,"10xsolar"]:
        vrbls[planet][exp][metallicity] = {}
        # Cannot have same time frame until all runs are finished then just specify t in parameters and remove if statements
        if metallicity == "solar":
            t=99
            day = (t*10)+10
        if metallicity == "10xsolar":
            t=44
            day = (t*10)+10
        
        # Load data
        fpath = SUITES[planet][exp][metallicity]["dir_for_merged"]
        fname = f"{SUITES[planet][exp][metallicity]['suite']}.nc"
        u = iris.load_cube(str(fpath / fname), um.u)
        v = iris.load_cube(str(fpath / fname), um.v)
        w = iris.load_cube(str(fpath / fname), um.w)
        pres = iris.load_cube(str(fpath / fname), "air_pressure")
        temp = iris.load_cube(str(fpath / fname), "air_temperature")
        
        # Calculate density for each metallicity case
        if metallicity == "solar":
            spc = spc_HD209
        if metallicity == "10xsolar":
            spc = spc_mdh1
    
        density = pres/(spc*temp)
    
        # fixes gridding bug by removing var_name from the cubes
        for cube in [u, v, w, density]:
            for coord_name in [um.z, um.x, um.y]:  # or um.y
                cube.coord(coord_name).var_name = coord_name  # or None
    
        # Pressure/temperature are on a height grid (in m) with different grid coordinate values to the height grid (in m) that the winds are on 
        # so need to regrid so they have the same grid values and can then be multiplied for the averages
        u_regrid = regrid_3d(u,density,um)
        v_regrid = regrid_3d(v,density,um)
    
        #slice at certain time
        u_space = u_regrid[t,:,:,:]
        v_space = v_regrid[t,:,:,:]
        w_space = w[t,:,:,:]
        rho_space = density[t,:,:,:]
        
        # extract coordinates for plotting
        lat_plt = w.coord("latitude").points
        lvl_plt = w.coord("level_height").points
        
        # extract coordinates for maths
        u_lvl = u_space[:,45,72]
        radius = coord_to_cube(
                    u_lvl, # wind cube with z in height 
                    um.z, # extract z in heigth
                ) + 9e7 #1.03e7 # add height_domain
        
        u_lat = u_space[33,:,72]
        lat_cube = coord_to_cube(
                  u_lat, # wind cube with lats in degrees 
                  um.y, # extract latitudes
                )
        
        print(np.shape(lat_cube))
        
        lat_cos_cube = apply_ufunc(np.cos, apply_ufunc(np.deg2rad, lat_cube))

        lat_cos_sq = lat_cos_cube ** 2

        # u manipulation
        u_zn = u_space.collapsed(["longitude"], iris.analysis.MEAN) # zonal mean of u
        u_prime = u_space - u_zn # perturbation of u

        # v manipulation
        v_zn = v_space.collapsed(["longitude"], iris.analysis.MEAN)
        v_prime = v_space - v_zn

        # w manipulation
        w_zn = w_space.collapsed(["longitude"], iris.analysis.MEAN)
        w_prime = w_space - w_zn

        # density manipulation
        rho_zn = rho_space.collapsed(["longitude"], iris.analysis.MEAN)

        # density v manipulation
        rho_v = rho_space * v_space
        rho_v_zn = rho_v.collapsed(["longitude"], iris.analysis.MEAN)
        rho_v_prime = rho_v - rho_v_zn

        # density w manipulation
        rho_w = rho_space * w_space
        rho_w_zn = rho_w.collapsed(["longitude"], iris.analysis.MEAN)
        rho_w_prime = rho_w - rho_w_zn

        # Mean Horizontal
        def mean_horizontal():
            deriv_arg = rho_v_zn * u_zn * lat_cos_sq
            numerator = -1 * d_dphi(deriv_arg)
            denominator = lat_cos_sq
            meridonal_momentum = divide(numerator,denominator)
            meridonal_momentum.units = "kg.s-4.K-1"
            return meridonal_momentum
        
        meridonal_mean_momentum = mean_horizontal()    

        # Mean Vertical
        def mean_vertical():
            deriv_arg = rho_w_zn * u_zn * radius ** 3
            numerator = -1 *deriv(deriv_arg, um.z)
            vertical_momentum = divide(numerator, radius ** 3)
            return vertical_momentum
    
        vertical_mean_momentum = mean_vertical()

        total_momentum = meridonal_mean_momentum + vertical_mean_momentum

        # Eddy Horizontal
        def eddy_horizontal():
            deriv_arg = (rho_v_prime * u_prime).collapsed(["longitude"], iris.analysis.MEAN) * lat_cos_sq
            numerator = -1 *d_dphi(deriv_arg)
            denominator = lat_cos_sq
            meridonal_eddy = divide(numerator,denominator)
            meridonal_eddy.units = "kg.s-4.K-1"
            return meridonal_eddy
    
        meridonal_mean_eddy = eddy_horizontal()

        # Eddy Vertical
        def eddy_vertical():
            deriv_arg = (rho_w_prime * u_prime).collapsed(["longitude"], iris.analysis.MEAN) * (radius ** 3)
            numerator = -1 * deriv(deriv_arg, um.z)
            vertical_eddy = divide(numerator, radius ** 3)
            return vertical_eddy

        vertical_mean_eddy = eddy_vertical()

        total_eddy = meridonal_mean_eddy + vertical_mean_eddy
        
        vrbls[planet][exp][metallicity]["momentum"] = {
            "meridonal" : meridonal_mean_momentum,
            "vertical" : vertical_mean_momentum,
            "total" : total_momentum,
        }
        
        vrbls[planet][exp][metallicity]["eddy"] = {
            "meridonal" : meridonal_mean_eddy,
            "vertical" : vertical_mean_eddy,
            "total" : total_eddy,
        }

for metallicity in ["solar"]: #,"10xsolar"]:

    # selects the names of the output files
    if metallicity == "solar":
        mdh = "MdH0"
        t=99
        day = (t*10)+10
    if metallicity == "10xsolar":
        mdh = "MdH1"
        t=44
        day = (t*10)+10
        
    for moment in ["momentum","eddy"]:
        fig, axes = plt.subplots(
            ncols=2, nrows=3, figsize=(15, 20), sharex=True, sharey=True, constrained_layout=True
        )
        #axes[0,0].invert_yaxis()
        iletters = subplot_label_generator()
        for exp, axcol in zip(["equilibrium", "kinetics"], axes.T):
            for comp, ax in zip(["meridonal","vertical","total"], axcol):
            
                # To normalise plots need to find the max and minimum of each data set then choose the larger magnitude and set vmax = -vmin or vmin = -vmax depending on which has a larger magnitude
                data_max = vrbls[planet][exp][metallicity][moment][comp].data.max()
                data_min = vrbls[planet][exp][metallicity][moment][comp].data.min()
            
                if abs(data_max) >= abs(data_min):
                    norm_max = data_max
                    norm_min = -data_max
                else:
                    norm_max = -data_min
                    norm_min = data_min
        
                p0 = ax.pcolormesh(
                    lat_plt,
                    lvl_plt,
                    vrbls[planet][exp][metallicity][moment][comp].data,
                    norm = mpl.colors.Normalize(vmin=norm_min, vmax=norm_max),
                    cmap="RdBu_r",
                    shading="auto",
                    rasterized=True,
                )
            
                p1 = ax.contour(
                    lat_plt,
                    lvl_plt,
                    vrbls[planet][exp][metallicity][moment][comp].data,
                    levels=np.arange(-2, 8, 2),
                    colors="k",
                    linewidths=0.5,
                )

                ax.clabel(p1, p1.levels, inline=True, fmt="%.0f", fontsize="small")

                fig.colorbar(
                    p0,
                    ax=ax,
                    label = "Hi",
                    orientation="horizontal",
                    #shrink=0.3,
                )
    
                ax.set_title(f"{planet} {comp} {moment} budget for {metallicity} metallicity {exp} (day {day})")
                ax.set_ylabel("Height [m]")
                ax.set_xlabel("Latitude")
                figname = f"Denis_{planet}_{mdh}_{moment}_{day}.png"
                
                if limit == 0:
                    ax.set_xlim(40, -40)
                    ax.set_ylim(0.2e7,0.9e7)
                    figname = f"Denis_{planet}_{mdh}_{moment}_{day}_limited.png"
                
                fig.savefig(output_dir / figname)