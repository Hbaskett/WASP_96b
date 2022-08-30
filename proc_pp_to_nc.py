import warnings
from pathlib import Path
import iris
import numpy as np

from aeolus.coord import coord_to_cube, regrid_3d, replace_z_coord, volume_weights_cube
from aeolus.model import um
from util_commons import STASH, SUITES

warnings.filterwarnings("ignore", module="iris")

# Select 
planet = "wasp96b"
exp = "kinetics"
metallicity = "10xsolar"
fname = f"{SUITES[planet][exp][metallicity]['suite']}.nc"

npath = SUITES[planet][exp][metallicity]["dir_for_raw"]
flist = []
for file in npath.glob("**/atmos_base_*/atmosa.p*"):
  if (
      "atmosa.pb" in file.name
      or "atmosa.pc" in file.name
      or "atmosa.pd" in file.name
      or "atmosa.pe" in file.name
      or "atmosa.pf" in file.name
      or "atmosa.pg" in file.name  # skip atmosa.ph* for now
  ):
      flist.append(str(file))
      
# Print sorted list of raw files
# len(sorted(flist))
# Load raw data
cl = iris.load(flist, STASH.keys())  # 38 fields
#print(cl)

for key, value in STASH.items():
    for cube in cl:
        if key == cube.name() and "mole fraction" in value:
            cube.rename(value)
            cube.units = "1"
            
# Convert PosixPaths to strings to be able to save them to .nc as metadata
for key, value in SUITES[planet][exp][metallicity].items():
    if "dir" in key:
        SUITES[planet][exp][metallicity][key] = str(value)

cl_on_hlevs = iris.cube.CubeList()
for cube in cl:
    # Remove the lowest model_level_number to match other variables
    if cube.name() in ["air_pressure", "upward_air_velocity"]:
        new_cube = cube[:, 1:, ...]
    else:
        new_cube = cube
    # Replace model_level_number by level_height as a dimensional vertical coordinate
    if "model_level_number" in [coord.name() for coord in new_cube.dim_coords]:
        new_cube = replace_z_coord(new_cube)
    # Add metadata
    new_cube.attributes.update(SUITES[planet][exp][metallicity])
    cl_on_hlevs.append(new_cube)
    
# Save to .nc
iris.save(
    cl_on_hlevs,
    str(Path(SUITES[planet][exp][metallicity]["dir_for_merged"]) / fname),
)

print("Done")