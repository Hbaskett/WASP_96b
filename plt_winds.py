import warnings
from pathlib import Path
import numpy as np
import iris
import iris.pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
from aeolus.coord import interp_cube_from_height_to_pressure_levels
from aeolus.model import um
from aeolus.plot import add_custom_legend, subplot_label_generator
from util_commons import GASES, PLANETS, SUITES

#Parameters
warnings.filterwarnings("ignore", module="iris")
planet = "wasp96b"
exp = "equilibrium"
metallicity = "solar"
mdh = "MdH0"
x = "lons"
y = "lats"
direction = "longitude"
t = 0 # time slice integer
tgt_plevs = np.logspace(7, 2, 65)  # target pressure levels

# Process data
vrbls = {}
vrbls[planet] = {}
for exp in ["equilibrium", "kinetics"]:
    vrbls[planet][exp] = {}
    for metallicity in ["solar","10xsolar"]:
        vrbls[planet][exp][metallicity] = {}
        
        # Cannot have same time frame until all runs are finished then just specify t in parameters and remove if statements
        if metallicity == "solar":
            time_step = np.arange(0,100,1)
            #print(time_step)
        if metallicity == "10xsolar":
            time_step = np.arange(0,70,1)
        
        for t in time_step:
            vrbls[planet][exp][metallicity][t] = {}
            # Load data
            fpath = SUITES[planet][exp][metallicity]["dir_for_merged"]
            fname = f"{SUITES[planet][exp][metallicity]['suite']}.nc"
            u = iris.load_cube(str(fpath / fname), um.u)
            v = iris.load_cube(str(fpath / fname), um.v)
            w = iris.load_cube(str(fpath / fname), um.w) # remove scaled for plotting?
            pres = iris.load_cube(str(fpath / fname), "air_pressure")
    
            #slice at certain time and remove extra level from pressure
            u_space = u[t,:,:,:]
            v_space = v[t,:,0:90,:]
            w_space = w[t,:,:,:]
            pres_space = pres[t,0:66,:,:]
    
            #interpolate u wind speed to pressure levels
            u_plevs = interp_cube_from_height_to_pressure_levels(u_space, pres_space, tgt_plevs)
            v_plevs = interp_cube_from_height_to_pressure_levels(v_space, pres_space, tgt_plevs)
            w_plevs = interp_cube_from_height_to_pressure_levels(w_space, pres_space, tgt_plevs)
            pres_coord = u_plevs.coord("air_pressure")
    
            #extract space coordinates
            lons = u_plevs.coord("longitude").points
            lats = u_plevs.coord("latitude").points
            lvl = u_plevs.coord("air_pressure").points
    
            for direction in ["longitude","latitude","air_pressure"]:
                #calculate zonal mean zonal wind speed
                u_plevs_znl_mean = u_plevs.collapsed(["time", direction], iris.analysis.MEAN)
                v_plevs_znl_mean = v_plevs.collapsed(["time", direction], iris.analysis.MEAN)
                w_plevs_znl_mean = w_plevs.collapsed(["time", direction], iris.analysis.MEAN)
                u_plevs_znl_mean.convert_units("km s-1")
                v_plevs_znl_mean.convert_units("km s-1")
                w_plevs_znl_mean.convert_units("km s-1")
        
                #find max speed direction for each case
                max_u_z, max_u_y = np.where(u_plevs_znl_mean.data == u_plevs_znl_mean.data.max())
                max_v_x, max_v_z = np.where(v_plevs_znl_mean.data == v_plevs_znl_mean.data.max())
                max_w_x, max_w_y = np.where(w_plevs_znl_mean.data == w_plevs_znl_mean.data.max())
    
                # Assemble data
                vrbls[planet][exp][metallicity][t][direction] = {
                    "zonal" : u_plevs_znl_mean,
                    "meridonal" : v_plevs_znl_mean,
                    "vertical" : w_plevs_znl_mean,
                    "max_u_z" : max_u_z,
                    "max_u_y" : max_u_y,
                    "max_v_x" : max_v_x,
                    "max_v_z" : max_v_z,
                    "max_w_x" : max_w_x,
                    "max_w_y" : max_w_y,
                }

for metallicity in ["solar","10xsolar"]:

    # selects the names of the output files
    if metallicity == "solar":
        mdh = "MdH0"
        time_step = np.arange(0,100,1)
    if metallicity == "10xsolar":
        mdh = "MdH1"
        time_step = np.arange(0,70,1)
        
    for t in time_step:
        day = (t*10)+10
    
        for direction in ["longitude","latitude","air_pressure"]:
            for wind in ["zonal","meridonal","vertical"]:
                fig, axes = plt.subplots(
                    ncols=2, nrows=1, figsize=(14, 8), sharex=True, sharey=True, constrained_layout=True
                )
            
                #Only invert when pressure on y axis strange notation but code is correct
                if direction != "air_pressure":
                    axes[0].invert_yaxis()
            
                iletters = subplot_label_generator()
                for exp, ax in zip(["equilibrium", "kinetics"], axes.T):
                    #Need to match dimensions of data vs axis arguments think possible via array but using 9 conditionals one for each circumstance (atm)
                    if (direction == "longitude" and wind == "zonal") or (direction == "longitude" and wind == "meridonal") or (direction == "longitude" and wind == "vertical"):
                        x = lats
                        y = pres_coord.points
                        ax.set_yscale("log")
                        ax.set_ylabel("Pressure [Pa]")
                        ax.set_xlabel("Latitude")
            
                    if (direction == "latitude" and wind == "zonal") or (direction == "latitude" and wind == "meridonal") or (direction == "latitude" and wind == "vertical"):
                        x = lons
                        y = pres_coord.points
                        ax.set_yscale("log")
                        ax.set_ylabel("Pressure [Pa]")
                        ax.set_xlabel("Longitude")
                
                    if (direction == "air_pressure" and wind == "zonal") or (direction == "air_pressure" and wind == "meridonal") or (direction == "air_pressure" and wind == "vertical"):
                        x = lons
                        y = lats
                        ax.set_ylabel("Latitude")
                        ax.set_xlabel("Longitude")
                
                    # To normalise plots need to find the max and minimum of each data set then choose the larger magnitude and set vmax = -vmin or vmin = -vmax depending on which has a larger magnitude
                    data_max = vrbls[planet][exp][metallicity][t][direction][wind].data.max()
                    data_min = vrbls[planet][exp][metallicity][t][direction][wind].data.min()
            
                    if abs(data_max) >= abs(data_min):
                        norm_max = data_max
                        norm_min = -data_max
                    else:
                        norm_max = -data_min
                        norm_min = data_min
            
                    # Plot all contours of each wind type
                    p0 = ax.pcolormesh(
                        x,
                        y,
                        vrbls[planet][exp][metallicity][t][direction][wind].data,
                        norm = mpl.colors.Normalize(vmin=norm_min, vmax=norm_max),
                        cmap="RdBu_r",
                        shading="auto",
                        rasterized=True,
                    ) 
            
                    p1 = ax.contour(
                        x,
                        y,
                        vrbls[planet][exp][metallicity][t][direction][wind].data,
                        levels=np.arange(-2, 8, 2),
                        colors="k",
                        linewidths=1,
                    )

                    p2 = ax.contour(
                        x,
                        y,
                        vrbls[planet][exp][metallicity][t][direction][wind].data,
                        levels=[l for l in np.arange(-2, 0.25, 0.25) if l != 0.0],
                        colors="k",
                        linewidths=0.5,
                    )

                    ax.clabel(p1, p1.levels, inline=True, fmt="%.0f", fontsize="small")
                    ax.clabel(p2, p2.levels, inline=True, fmt="%.2f", fontsize="small")
            
                    # Correctly titling figures and giving save files usable names
                    if direction == "air_pressure" :
                        comp_axis = "vertical"
                        ax.set_title(f"{comp_axis} mean {wind} wind for {metallicity} metallicity {exp} (day {day})")
                        fig.colorbar(
                            p0,
                            ax=ax,
                            label = f"{wind} wind velocity in [km $s^{-1}$]",
                            orientation="horizontal",
                            #shrink=0.3,
                        )
                        figname = f"{mdh}_{comp_axis}_mean_{wind}_wind_{day}.png"
                    if direction == "longitude":
                        comp_axis = "zonal"
                        ax.set_title(f"{comp_axis} mean {wind} wind for {metallicity} metallicity {exp} (day {day})")
                        fig.colorbar(
                            p0,
                            ax=ax,
                            label = f"{wind} wind velocity in [km $s^{-1}$]",
                            orientation="horizontal",
                            #shrink=0.3,
                        )
                        figname = f"{mdh}_{comp_axis}_mean_{wind}_wind_{day}.png"
                    if direction == "latitude":
                        comp_axis = "meridonal"
                        ax.set_title(f"{comp_axis} mean {wind} wind for {metallicity} metallicity {exp} (day {day})")
                        fig.colorbar(
                            p0,
                            ax=ax,
                            label = f"{wind} wind velocity in [km $s^{-1}$]",
                            orientation="horizontal",
                            #shrink=0.3,
                        )
                        figname = f"{mdh}_{comp_axis}_mean_{wind}_wind_{day}.png" 
                    
                    output_dir = Path.home() / "um" / "um_runs" / "winds" / mdh / f"day_{day}"  
            
                    fig.savefig(output_dir / figname) 
                    fig.close()