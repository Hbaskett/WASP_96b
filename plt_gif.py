import imageio
import numpy as np
from pathlib import Path
import os

# Used to make new directories for plots
#for x in np.arange(10,1010,10):
#    parent_dir = Path.home() / "um" / "um_runs" / "winds" / "MdH1"   
#    directory = f"day_{x}"
#    path = os.path.join(parent_dir, directory)
#    os.mkdir(path)

select = 1 # use this to select which gif you want to make 
# 0 - wind contours
# 1 PT structures
# 2 species vertical profiles

# make gifs

images_MdH0 = []
images_MdH1 = []

for mdh in ["MdH0","MdH1"]:
    
    if mdh == "MdH0":
        day_step = np.arange(10,1010,10)
        images = images_MdH0
    if mdh == "MdH1":
        day_step = np.arange(10,710,10)
        images = images_MdH1
        
    if select == 0:
        
        for x in ["zonal","meridonal","vertical"]:
            for y in ["zonal","meridonal","vertical"]:
            
                images.clear()
                for day in day_step:
        
                    # Choose images to compile
                    image_dir = Path.home() / "um" / "um_runs" / "winds" / mdh / f"day_{day}"
                    image_file = f"{mdh}_{x}_mean_{y}_wind_{day}.png"
        
                    # Choose name and output directory of final gif
                    gif_dir = Path.home() / "um" / "um_runs" / "winds" 
                    gif_file = f"{mdh}_{x}_mean_{y}_wind_total.gif"
                    images.append(imageio.imread(image_dir / image_file))
        
                imageio.mimsave(gif_dir / gif_file, images)
                
    if select == 1:
    
        images.clear()
        for day in day_step:
        
            # Choose images to compile
            image_dir = Path.home() / "um" / "um_runs" / "PT_lats" / mdh 
            image_file = f"{mdh}_PT_{day}.png"
        
            # Choose name and output directory of final gif
            gif_dir = Path.home() / "um" / "um_runs" / "PT_lats" / mdh 
            gif_file = f"{mdh}_PT_total.gif"
            images.append(imageio.imread(image_dir / image_file))
        
        imageio.mimsave(gif_dir / gif_file, images)
