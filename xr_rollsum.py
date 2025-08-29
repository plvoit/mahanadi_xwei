'''
Paul Voit 29 Aug 2025
This script finds the year maxima for every duration and stores them in NetCDF.
These files are then used to fit the dGEV in the next step.
'''

import xarray as xr
import sys
import os

path = "/path/to/your/workdir"
location = sys.argv[1]
nc = xr.open_dataset(f"{path}input/rainfall_ncdf/Rainfall_{location}.nc")
nc = nc.Rainfall

#@Saikat: Adjust the durations here. They have to be the same as in the dGEV-script and the other scripts
durations = [1, 3, 6, 30, 60, 90]
#@Saikat: Decide here how many NAs you want to accept. Otherwise the whole series could turn to NA, when you
#do the aggregation
na_tolerance = [1, 3, 5, 27, 54, 81] #name is confusing, maximum amount of NAs for each duration

# his aligns the window to the right, so the 30-day sum is computed for the current day and the preceding 29 days,
# placing the result at the end of the window. CDO should also do it like this

if not os.path.exists(f"{path}output/runsum/{location}"):
    os.mkdir(f"{path}output/runsum/{location}")

if not os.path.exists(f"{path}output/yearmax/{location}"):
    os.mkdir(f"{path}output/yearmax/{location}")

for  counter, d in enumerate(durations):
    print(counter)

    if durations[counter] == 1:
        rollsum = nc.round(1)

    else:
        rollsum = nc.rolling(time=d, min_periods=na_tolerance[counter], center=False).sum()
        rollsum = rollsum.round(1)

    rollsum.to_netcdf(f"{path}output/runsum/{location}/{str(durations[counter]).zfill(2)}d_runsum.nc")
    yearmax = rollsum.resample(time="YE").max(skipna=True)
    yearmax.to_netcdf(f"{path}output/yearmax/{location}/{str(durations[counter]).zfill(2)}d_yearmax.nc")

