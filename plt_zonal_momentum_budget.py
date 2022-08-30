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
comp = "meridonal"
moment = "momentum"
limit = 0 # Use 0 to plot zoomed data set at jet or full data set
t = 0 # time slice integer
day = (t*10)+10 # calculate the day to put into the figures titles
output_dir = Path.home() / "um" / "um_runs" / "check_conservation" / "zonal_momentum_budget"

# Process data
vrbls = {}
vrbls[planet] = {}
for exp in ["equilibrium", "kinetics"]:
    vrbls[planet][exp] = {}
    for metallicity in ["solar","10xsolar"]:
        vrbls[planet][exp][metallicity] = {}
        
        # Cannot have same time frame until all runs are finished then just specify t in parameters
        if metallicity == "solar"
            spc = spc_mdh0
        if metallicity == "10xsolar":
            spc = spc_mdh1
        
        # Load data and slice at a give t
        fpath = SUITES[planet][exp][metallicity]["dir_for_merged"]
        fname = f"{SUITES[planet][exp][metallicity]['suite']}.nc"
        u = iris.load_cube(str(fpath / fname), um.u)[t,:,:,:]
        v = iris.load_cube(str(fpath / fname), um.v)[t,:,:,:]
        w = iris.load_cube(str(fpath / fname), um.w)[t,:,:,:]
        pres = iris.load_cube(str(fpath / fname), "air_pressure")[t,:,:,:]
        temp = iris.load_cube(str(fpath / fname), "air_temperature")[t,:,:,:]
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

for metallicity in ["solar","10xsolar"]:

    # selects the names of the output files
    if metallicity == "solar":
        mdh = "MdH0"
    if metallicity == "10xsolar":
        mdh = "MdH1"
       
    for moment in ["momentum","eddy"]:
        fig, axes = plt.subplots(
            ncols=2, nrows=3, figsize=(24, 30), sharex=True, sharey=True, constrained_layout=True
        )
        #axes[0,0].invert_yaxis()
        iletters = subplot_label_generator()
        for exp, axcol in zip(["equilibrium", "kinetics"], axes.T):
            for comp, ax in zip(["meridonal","vertical","total"], axcol):
            
                p0 = ax.contourf(
                    lats,
                    lvl,
                    vrbls[planet][exp][metallicity][moment][comp].data,
                    levels=100, # argument which selects how many steps the colorbar is split into
                    norm = mpl.colors.TwoSlopeNorm(vcenter = 0), # keeps 0 as white on contour
                    cmap="RdBu_r",
                    extend = 'both',
                    alpha = 1,
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
                    #label = "Hi", # Don't know what to label the colorbar
                    orientation="horizontal",
                    #shrink=0.3,
                    extend = "both"
                )
                
                # Set titles
                if moment == "momentum":
                    if comp == "meridonal":
                        ax.set_title(f"{planet} Latitudinal gradient in the meridonal mean flow momentum flux for {metallicity} metallicity {exp} (day {day})", fontsize = 15)
                    if comp == "vertical":
                        ax.set_title(f"{planet} Vertical gradient in the vertical mean flow momentum flux for {metallicity} metallicity {exp} (day {day})", fontsize = 15)
                    if comp == "total":
                        ax.set_title(f"{planet} Sum of above components for {metallicity} metallicity {exp} (day {day})", fontsize = 20)
                
                if moment == "eddy":
                    if comp == "meridonal":
                        ax.set_title(f"{planet} Latitudinal gradient in the meridonal eddy momentum flux for {metallicity} metallicity {exp} (day {day})", fontsize = 15)
                    if comp == "vertical":
                        ax.set_title(f"{planet} Vertical gradient in the vertical eddy momentum flux for {metallicity} metallicity {exp} (day {day})", fontsize = 15)
                    if comp == "total":
                        ax.set_title(f"{planet} Sum of above components for {metallicity} metallicity {exp} (day {day})", fontsize = 20)
                        
                ax.set_ylabel("Height [m]", fontsize = 20)
                ax.set_xlabel("Latitude", fontsize = 20)
                figname = f"{planet}_{mdh}_{moment}_{day}.png"
                
                if limit == 0:
                    ax.set_xlim(40, -40)
                    ax.set_ylim(0.2e7,0.9e7)
                    figname = f"{planet}_{mdh}_{moment}_{day}_limited.png"
                
                fig.savefig(output_dir / figname) 
