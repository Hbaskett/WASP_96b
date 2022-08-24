import warnings
from pathlib import Path
import numpy as np
import scipy as sp
import iris
import iris.pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
from iris.analysis.maths import apply_ufunc
from aeolus.calc import deriv
from aeolus.coord import interp_cube_from_height_to_pressure_levels, regrid_3d
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
limit = 0 # Use 0 to plot zoomed data set at jet or full data set
t = 0 # time slice integer
day = (t*10)+10
tgt_plevs = np.logspace(7, 2, 65)  # target pressure levels
output_dir = Path.home() / "um" / "um_runs" / "check_conservation" 

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
    
        #slice at certain time and remove extra levels
        u_space = u_regrid[t,:,:,:]
        v_space = v_regrid[t,:,0:90,:]
        w_space = w[t,:,:,:]
        density_space = density[t,:,:,:]

        #extract space coordinates
        lons = w.coord("longitude").points # think this coordinate is redundant
        lats = w.coord("latitude").points
        lvl = w.coord("level_height").points
        radius = w.coord("level_height").points + 9e7 #1.03e7 #add the inner radius to make up the full radius at each point 
        
        # Equations from Mayne 2017 section 3.3.2
        # calculate parts of the equations 
        pwu_znl_mean = (density_space*w_space*u_space).collapsed(["longitude"], iris.analysis.MEAN)
        pvu_znl_mean = (density_space*v_space*u_space).collapsed(["longitude"], iris.analysis.MEAN)
        pw_znl_mean = (density_space*w_space).collapsed(["longitude"], iris.analysis.MEAN)
        pv_znl_mean = (density_space*v_space).collapsed(["longitude"], iris.analysis.MEAN)
        u_znl_mean = u_space.collapsed(["longitude"], iris.analysis.MEAN)
        # Using theory in Appendix C to calculate perturbations
        prime_w_znl_mean = pwu_znl_mean - (pw_znl_mean*u_znl_mean)
        prime_v_znl_mean = pvu_znl_mean - (pv_znl_mean*u_znl_mean)
        # find cos squared of latitude - convert from degrees to radians for np.cos
        cos_lats_sqr = np.cos(lats*(np.pi/180))**2
        # give another dimension to lvl array
        radius_2d = np.atleast_2d(radius**3)
        # transpose the arry from a column to a row
        radius_row = np.transpose(radius_2d)
        
        # Mean meridonal momentum flux
        A = pv_znl_mean.data*u_znl_mean.data*cos_lats_sqr
        grad_A = np.gradient(A, axis=1)
        grad_lats = np.gradient(lats*(np.pi/180), axis=0)
        grad_ratio_A = (1/grad_lats)*grad_A
        meridonal_momentum = - grad_ratio_A*(1/(np.outer(lvl,cos_lats_sqr)))

        # Mean vertical momentum flux
        B = radius_row*pw_znl_mean.data*u_znl_mean.data
        grad_B = np.gradient(B, axis=0)
        grad_radius = np.gradient(radius, axis=0)
        grad_radius_2d = np.atleast_2d(grad_radius)
        grad_radius_row = np.transpose(grad_radius_2d)
        grad_ratio_B = (1/grad_radius_row)*grad_B 
        vertical_momentum = -grad_ratio_B*(1/radius_row)
    
        total_momentum = meridonal_momentum + vertical_momentum
    
        vrbls[planet][exp][metallicity]["momentum"] = {
            "meridonal" : meridonal_momentum,
            "vertical" : vertical_momentum,
            "total" : total_momentum,
        }
       
        # Mean meridoanl eddy momentum flux
        C = prime_v_znl_mean.data*cos_lats_sqr
        grad_C = np.gradient(C, axis=1)
        grad_ratio_C = (1/grad_lats)*grad_C
        meridonal_eddy = - grad_ratio_C*(1/(np.outer(lvl,cos_lats_sqr)))

        # Mean vertical eddy momentum flux
        D = radius_row*prime_w_znl_mean.data
        grad_D = np.gradient(D, axis=0)
        grad_ratio_D = (1/grad_radius_row)*grad_D
        vertical_eddy = - grad_ratio_D*(1/radius_row)
    
        total_eddy = meridonal_eddy + vertical_eddy

        vrbls[planet][exp][metallicity]["eddy"] = {
            "meridonal" : meridonal_eddy,
            "vertical" : vertical_eddy,
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
                    lats,
                    lvl,
                    vrbls[planet][exp][metallicity][moment][comp].data,
                    norm = mpl.colors.Normalize(vmin=norm_min, vmax=norm_max),
                    cmap="RdBu_r",
                    shading="auto",
                    rasterized=True,
                )
            
                p1 = ax.contour(
                    lats,
                    lvl,
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
                figname = f"{planet}_{mdh}_{moment}_{day}.png"
                
                if limit == 0:
                    ax.set_xlim(40, -40)
                    ax.set_ylim(0.2e7,0.9e7)
                    figname = f"{planet}_{mdh}_{moment}_{day}_limited.png"
                
                fig.savefig(output_dir / figname) 
