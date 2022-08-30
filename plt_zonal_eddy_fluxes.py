import warnings
from pathlib import Path
import numpy as np
import scipy as sp
import iris
import iris.pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
from aeolus.calc import deriv
from aeolus.coord import regrid_3d
from aeolus.model import um
from aeolus.plot import add_custom_legend, subplot_label_generator
from util_commons import GASES, PLANETS, SUITES
#from HD209_commons import SUITES

#Parameters
warnings.filterwarnings("ignore", module="iris")
planet = "wasp96b"
exp = "equilibrium"
metallicity = "solar"
mdh = "MdH0"
inner_r_wasp = 83893200.0 
spc_mdh0 = 3516.2947232694723
spc_mdh1 = 3164.69353854304
spc_HD209 =  3516.6
comp = "horizontal"
output_dir = Path.home() / "um" / "um_runs" / "check_conservation" / "zonal_eddy_fluxes"

# Process data
vrbls = {}
vrbls[planet] = {}
for exp in ["equilibrium", "kinetics"]:
    vrbls[planet][exp] = {}
    for metallicity in ["solar","10xsolar"]:
        vrbls[planet][exp][metallicity] = {}
        
        # Cannot have same time frame until all runs are finished then just specify t in parameters and remove if statements
        if metallicity == "solar":
            spc = spc_mdh0
        if metallicity == "10xsolar":
            spc = spc_mdh1
        
        # Load data and slice at a give t
        fpath = SUITES[planet][exp][metallicity]["dir_for_merged"]
        fname = f"{SUITES[planet][exp][metallicity]['suite']}.nc"
        u = iris.load_cube(str(fpath / fname), um.u)
        v = iris.load_cube(str(fpath / fname), um.v)
        w = iris.load_cube(str(fpath / fname), um.w)
        pres = iris.load_cube(str(fpath / fname), "air_pressure")
        temp = iris.load_cube(str(fpath / fname), "air_temperature")
        density = pres/(spc*temp)
        
        # fixes gridding bug by removing var_name from the cubes
        for cube in [u, v, w, density]:
            for coord_name in [um.z, um.x, um.y]:  # or um.y
                cube.coord(coord_name).var_name = coord_name  # or None
    
        # Pressure/temperature are on a height grid (in m) with different grid coordinate values to the height grid (in m) that the winds are on 
        # so need to regrid so they have the same grid values and can then be multiplied for the averages
        u_regrid = regrid_3d(u,density,um)
        v_regrid = regrid_3d(v,density,um)
        
        # load into maths variables
        u_space = u_regrid
        v_space = v_regrid
        w_space = w
        density_space = density
        
        #extract space coordinates
        lons = w.coord("longitude").points # think this coordinate is redundant
        lats = w.coord("latitude").points
        lvl = w.coord("level_height").points
        radius = w.coord("level_height").points + inner_r_wasp #1.03e7 #add the inner radius to make up the full radius at each point 
        
        #using combination of theory from Mayne 2017 and Showman 2015
        # Take time and zonal mean of each wind component
        u_znl_time_mean = (u_space).collapsed(['time', 'longitude'], iris.analysis.MEAN)
        v_znl_time_mean = (v_space).collapsed(['time', 'longitude'], iris.analysis.MEAN)
        w_znl_time_mean = (w_space).collapsed(['time', 'longitude'], iris.analysis.MEAN)
        
        #Take time and zonal mean of wind products
        uv_znl_time_mean = (u_space*v_space).collapsed(['time', 'longitude'], iris.analysis.MEAN)
        uw_znl_time_mean = (u_space*w_space).collapsed(['time', 'longitude'], iris.analysis.MEAN)
        
        #Use perturbation theory to calculate eddy fluxes (primes)
        horizontal_eddy_flux = uv_znl_time_mean - (u_znl_time_mean * v_znl_time_mean)
        vertical_eddy_flux = uw_znl_time_mean - (u_znl_time_mean * w_znl_time_mean)
        
        # find cos of latitude - convert from degrees to radians for np.cos
        #cos_lats = np.cos(lats*(np.pi/180))
        # give another dimension to lvl array
        #radius_2d = np.atleast_2d(radius**3)
        # transpose the arry from a column to a row
        #radius_row = np.transpose(radius_2d)
        
        vrbls[planet][exp][metallicity]["eddy"] = {
            "horizontal" : horizontal_eddy_flux,
            "vertical" : vertical_eddy_flux,
        }
        
for metallicity in ["solar","10xsolar"]:

    # selects the names of the output files
    if metallicity == "solar":
        mdh = "MdH0"
    if metallicity == "10xsolar":
        mdh = "MdH1"
        
    fig, axes = plt.subplots(
        ncols=2, nrows=2, figsize=(18, 15), sharex=True, sharey=True, constrained_layout=True
    )
    #axes[0,0].invert_yaxis()
    iletters = subplot_label_generator()
    for exp, axcol in zip(["equilibrium", "kinetics"], axes.T):
        for comp, ax in zip(["horizontal","vertical"], axcol):
            
            p0 = ax.contourf(
                lats,
                lvl,
                vrbls[planet][exp][metallicity]["eddy"][comp].data,
                levels=100,
                norm = mpl.colors.TwoSlopeNorm(vcenter = 0), #vmin=norm_min, vmax=norm_max),
                cmap="RdBu_r",
                extend = 'both',
                alpha = 1,
                #shading="auto",
                #rasterized=True,
            )
            
            p1 = ax.contour(
                lats,
                lvl,
                vrbls[planet][exp][metallicity]["eddy"][comp].data,
                levels=np.arange(-2, 8, 2),
                colors="k",
                linewidths=0.5,
            )

            ax.clabel(p1, p1.levels, inline=True, fmt="%.0f", fontsize="small")

            fig.colorbar(
                p0,
                ax=ax,
                #label = "Hi",
                orientation="horizontal",
                #shrink=0.3,
                extend = "both"
            )
                
            # Set titles
            if comp == "horizontal":
                ax.set_title(f"{planet} Time averaged zonal mean horizontal eddy flux for {metallicity} metallicity {exp}", fontsize = 10)
            if comp == "vertical":
                ax.set_title(f"{planet} Time averaged zonal mean vertical eddy flux for {metallicity} metallicity {exp}", fontsize = 10)
                    
            ax.set_ylabel("Height [m]", fontsize = 10)
            ax.set_xlabel("Latitude", fontsize = 10)
            figname = f"{planet}_{mdh}_eddy_flux.png"
                
            fig.savefig(output_dir / figname) 
        