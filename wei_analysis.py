'''
Paul Voit 16 Oct 2024
We want to know how which duration where extreme for the triggering event.
For now we will just analyse the 90 das prior to the landslide with the WEI

@Saikat: So here we basically look into the past from the date were a landslide happened. In our case this would
be the date of the flood peak. These dates are stored in a dataframe. I'll attach an example dataframe so you can
format the flood info accordingly.
'''

import os
import sys
import numpy as np
import xarray as xr
import shutil
from scipy.stats import genextreme as gev_dist
import pandas as pd
import xwei_functions as xf

location = sys.argv[1]

path = "/path_to_your_workdir"
os.chdir(path)

#@Saikat: We capped the return periods at 200 because our time series was so short.
#Maybe we can go a bit higher if we really have 120 years of data. I'd say maximum 500.
max_rp = 200
#@Saikat: Important needs to be changed to the cell size in your data set
cell_size = 30.98

#get the date of the event
info_df = pd.read_csv("Event_dates.csv", sep=",")
date = info_df.loc[info_df["Region"] == location, "Date_Event"].values[0]
date = pd.to_datetime(date, format="%d/%m/%Y")
start_date = (date - pd.Timedelta
(days=270)) #for the 90-day duration we look 269 days into the past

#subset the rainfall array
nc = xr.open_dataset(f"{path}input/rainfall_ncdf/Rainfall_{location}.nc")
nc = nc.Rainfall

if not os.path.exists(f"{path}output/eta_series"):
    os.mkdir(f"{path}output/eta_series")

event_nc = nc.sel(time=slice(start_date, date))

path2parameter_fits = f"{path}output/dgev_original/{location}"
dgev_parms = dict(
    mod_loc=np.genfromtxt(f"{path2parameter_fits}/mod_loc.csv", delimiter=",", ndmin=2),
    scale_0=np.genfromtxt(f"{path2parameter_fits}/scale_0.csv", delimiter=",", ndmin=2),
    shape=np.genfromtxt(f"{path2parameter_fits}/shape.csv", delimiter=",", ndmin=2),
    duration_offset=np.genfromtxt(f"{path2parameter_fits}/duration_offset.csv",
                               delimiter=",", ndmin=2),
    duration_exp=np.genfromtxt(f"{path2parameter_fits}/duration_exp.csv",
                            delimiter=",", ndmin=2))

#@Saikat: change
durations = [1, 3, 6, 30, 60, 90]

def create_dgev_parms_dict(durations):
    # pre-calculate the dgev parms
    for duration in durations:
        dgev_parms[f"sigma_{duration}"] = dgev_parms["scale_0"] * (
            int(duration) +
            dgev_parms["duration_offset"])**-dgev_parms["duration_exp"]
        dgev_parms[f"mu_{duration}"] = dgev_parms["mod_loc"] * dgev_parms[
            f"sigma_{duration}"]

create_dgev_parms_dict(durations)

#@Saikat: change
duration_levels=['01', '03', '06', '30', '60', '90']
nan_tolerance={'01': 1, '03': 3, '06': 5, '30': 27, '60': 54, '90': 81}


def eta_dgev(array, dgev_parms: dict, max_rp=1000,
         duration_levels=['01', '02', '04', '06', '12', '24', '48', '72'],
         tolerance_dict={'01': 1, '02': 2, '04': 4, '06': 6, '12': 11, '24': 22, '48': 36, '72': 60},
         cell_size_km2=1):
    """
    Function is adapted from wei.WeiClass.dgev. Its simplified and the rolling sums are done differently
    because I did not understand anymore why and how I did it in the original function.
    Here we use use different window sizes for each duration.
    Basically d * 3 into the past: For 1d we take the day and the two previous days, for 30d we take the day and 89
    previous dates etc...
    Here we look at all timestep with a rolling window, compute the Eta for each timestep and then extract the
    maximum Eta series for each duration.

    Calculation of return periods based on the concept of duration dependent GEV-curves. dGEV parameters were
    derived with the R-package IDF.
    References:
    Fauer, Felix S., et al. "Flexible and consistent quantile estimation for intensity–duration–frequency curves
    " Hydrology and Earth System Sciences 25.12 (2021): 6479-6494.

    Koutsoyiannis, D., Kozonis, D., and Manetas, A.: A mathematical framework for studying rainfall intensity-
    duration-frequency relationships, J. Hydrol., 206, 118–135, https://doi.org/10.1016/S0022-1694(98)00097-3,
    1998

    :param array:xarray.core.dataarray.DataArray of the events rainfall
    :param dgev_parms: dict, dictionary with all the dgev parameters for all durations. Created with create_dgev_parms_dict
    :param max_rp: int, return periods higher than this value will be capped
    :param duration_levels: list, list of duration levels in string format
    :param tolerance_dict: dict, minimum values that need to be non-NaN for rolling sum calculation.
    :param cell_size_km2: int, cell size of the data set to compute Eta correctly

    :return:
     max_vals: pd.Dataframe containing the maximum eta series for each duration
    """

    results_dict = dict()

    for i in range(0, len(duration_levels)):
        duration = duration_levels[i]
        print(f"Calculating WEI for duration {duration}h.")
        # -1 because xarray subsetting includes the boundaries (unlike numpy indexing)
        start_date_subset = date - pd.Timedelta(int(duration) * 3 - 1, "d")

        if duration_levels[i] == "01":
            rollsum_array = array.sel(time=slice(start_date_subset, date)).copy()

        else:
            rollsum_array = array.sel(time=slice(start_date_subset, date)).copy().rolling(time=int(duration),
                                                                                          min_periods=tolerance_dict[duration],
                                                                                          center=False).sum()

        rollsum_array = rollsum_array.values

        # dirty fix to fix nan
        rollsum_array[np.isnan(rollsum_array)] = 1
        Pu = gev_dist.cdf(rollsum_array / int(duration), c=dgev_parms["shape"], loc=dgev_parms[f"mu_{int(duration)}"],
                          scale=dgev_parms[f"sigma_{int(duration)}"])

        Pu[Pu == 1] = 0.999  # To fix the runtime error when division is by zero

        rp_array = 1 / (1 - Pu)

        result = xf._calc_eta(max_rp=max_rp, rp_array=rp_array, cell_size_km2=cell_size_km2)

        results_dict[duration] = result

    #select the timestep with maximum values in the series for each duration
    max_eta = xf.get_max_eta(results_dict)

    max_vals = pd.DataFrame(max_eta).round(2)

    return max_vals


max_eta_df = eta_dgev(event_nc, dgev_parms, max_rp=max_rp, duration_levels=duration_levels,
                      tolerance_dict=nan_tolerance, cell_size_km2=cell_size)

max_eta_df.to_csv(f"{path}output/eta_series/eta_{location}.csv", index=False)

#next step is to find the maximum Eta Value in the whole timeseries to scale the axis accordingly.
results_dict = dict()

for duration in duration_levels:
    print(duration)
    rollsum_array = xr.open_dataset(f"{path}output/runsum/{location}/{duration}d_runsum.nc")
    rollsum_array = rollsum_array.Rainfall.values
    Pu = gev_dist.cdf(rollsum_array / int(duration), c=dgev_parms["shape"], loc=dgev_parms[f"mu_{int(duration)}"],
                          scale=dgev_parms[f"sigma_{int(duration)}"])

    Pu[Pu == 1] = 0.999  # To fix the runtime error when division is by zero
    rp_array = 1 / (1 - Pu)
    result = xf._calc_eta(max_rp=max_rp, rp_array=rp_array, cell_size_km2=cell_size)

    results_dict[duration] = result

# select the timestep with maximum values in the series for each duration
max_eta = xf.get_max_eta(results_dict)
#max_eta = max_eta.max().max()
max_eta = max_eta.max()

df = pd.DataFrame(max_eta)
df = df.rename(columns={0: "Eta"})
#@Saikat: Change path here
df.to_csv(f"/home/voit/Radolan/maria_landslides/output/hist_max_etas/max_eta_{location}.csv")


