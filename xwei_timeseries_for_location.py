'''
Paul Voit 2 October 2024
'''

import os
import sys
import numpy as np
import xarray as xr
import shutil
from scipy.spatial import Delaunay
import pandas as pd
import xwei_functions as import dgev_xwei_for_array

location = sys.argv[1]

path = "/path/to/your/workdir/"

os.chdir(path)

if not os.path.exists(f'{path}input/weights'):
    os.mkdir(f'{path}input/weights')

if not os.path.exists(f"{path}output/tracking"):
    os.mkdir(f"{path}output/tracking")


print("read dgev parameters")
path2parameter_fits = f"{path}output/dgev_original/{location}"
dgev_parms = dict(
    mod_loc=np.genfromtxt(f"{path2parameter_fits}/mod_loc.csv", delimiter=",", ndmin=2),
    scale_0=np.genfromtxt(f"{path2parameter_fits}/scale_0.csv", delimiter=",", ndmin=2),
    shape=np.genfromtxt(f"{path2parameter_fits}/shape.csv", delimiter=",", ndmin=2),
    duration_offset=np.genfromtxt(f"{path2parameter_fits}/duration_offset.csv",
                               delimiter=",", ndmin=2),
    duration_exp=np.genfromtxt(f"{path2parameter_fits}/duration_exp.csv",
                            delimiter=",", ndmin=2))

#@Saikat: change the durations here as in the other scripts. Yes, that could also be handled a bit more elegant
#for very small arrays with just one
durations = [1, 3, 6, 30, 60, 90]

'''
We do not need a window here because we will calculate the xWEI for the whole spatial domain and for a 90 days length
moving window
'''

def create_dgev_parms_dict(durations):
    # pre-calculate the dgev parms
    for duration in durations:
        dgev_parms[f"sigma_{duration}"] = dgev_parms["scale_0"] * (
            int(duration) +
            dgev_parms["duration_offset"])**-dgev_parms["duration_exp"]
        dgev_parms[f"mu_{duration}"] = dgev_parms["mod_loc"] * dgev_parms[
            f"sigma_{duration}"]


create_dgev_parms_dict(durations)


print("Calculating weights for grid")
#@Saikat: These weights are used, when the xWEI is calculated. Because we do it for every day, we
# can reuse these weights and speed up the computation a lot.

def interp_weights(xyz, uvw, d=2):
    tri = Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

len_x = dgev_parms["duration_exp"].shape[0] * dgev_parms["duration_exp"].shape[1] # e.g. if you have a 200km * 200 km window

#this defines how fine the interpolation of the 3-dimensional xWEI surface is.
#@Saikat: I think you can leave the resolution like this
resolution = 1000

coords = np.ones((len_x * len(durations), 2))

x_coords = list(np.arange(1, len_x + 1)) * len(durations)

# from https://stackoverflow.com/questions/2449077/duplicate-each-member-in-a-list
y_coords = [val for val in durations for _ in (range(1, len_x + 1))]

coords[:, 0] = np.log(x_coords)
coords[:, 1] = np.log(y_coords)

### create the refined grid on which the values will be later interpolated
x_range = np.linspace(0, np.log(len_x), len_x)
y_range = np.linspace(0, np.log(durations[-1]), resolution)
grid_x, grid_y = np.meshgrid(x_range, y_range)

uv = np.zeros([grid_x.shape[0] * grid_x.shape[1], 2])
uv[:, 0] = grid_x.flatten()
uv[:, 1] = grid_y.flatten()

# interpolate the eta values to get a surface. These weights can be reused for every 200km x 200km box
vtx, wts = interp_weights(coords[:, :2], uv)


if not os.path.exists(f'{path}input/weights/{location}'):
    os.mkdir(f'{path}input/weights/{location}')

vtx = pd.DataFrame(vtx)
vtx.to_csv(f'{path}input/weights/{location}/vtx.csv', index=False, header=False)

wts = pd.DataFrame(wts)
wts.to_csv(f'{path}input/weights/{location}/wts.csv', index=False, header=False)

weights = (vtx, wts)

#open the rainfall netcdf
xr_all = xr.open_dataset(f'{path}input/rainfall_ncdf/Rainfall_{location}.nc')

#@Saikat: Of course this variable can have a different name in your data set
time_info = xr_all.coords["time"].values

data_length = len(time_info)

class WorkPackage:
    '''
    Paul Voit 15 Oct 2024: Adapted and changed from the function in xWEI tracking
    dgev_parms: Dictionary with pre-calculated parameters for the dgev function
    input_array: xr.Dataset Dataset with rainfall values for which the xwei should be computed
    weights: tupel(vtx, wts) arrays which store the vertices and weights for the surface interpolation of xwei
    max_duration: int, the maximum duration that should be considered. This is for subsetting the arrays.
    '''

    def __init__(self, dgev_parms, input_array, weights, max_counter, max_duration):
        self.dgev_params = dgev_parms
        self.input_array = input_array
        self.weights = weights
        self.max_duration = max_duration
        self.max_counter = max_counter

#@Saikat: Here are the durations again which need to be changed
def create_duration_array(input_array, durations=[1, 3, 6, 30, 60, 90]):
    '''
    Paul Voit 15 Oct 2024: Adapted from the function in xWEI tracking. Changed here how we
    Here the aggregation happens. The aggregation considers the day of interest and the previous days:
                      e.g. for duration 3d, its the actual day and the 2 days before
                                        10d, actual day and 9 days before....

    :param input_array: np.array, array to be aggregated. At minimum has to be first dimension (time) of last duration
        (72h normally)
    :param durations: list, list of durations that should be considered
    :return:
    '''

    slice_all_durations = np.zeros(
        (len(durations), input_array.shape[1], input_array.shape[2]))
    slice_all_durations[:] = np.nan

    for i, duration in enumerate(durations):
        if i == 0:
            slice_all_durations[i, :, :] = input_array[-1, :, :] #the "last" slice

        else:
            # np.nansum results in np.nan + np.nan = 0. This shouldn't be a problem though, because these
            # values won't be used later and shouldn't affect the result

            slice_all_durations[i, :, :] = np.nansum(
                input_array[-duration:, :, :], axis=0)

    return slice_all_durations


if os.path.exists(f"{path}output/tracking/{location}"):
    shutil.rmtree(f"{path}output/tracking/{location}")
    os.mkdir(f"{path}output/tracking/{location}")
else:
    os.mkdir(f"{path}output/tracking/{location}")


def calc_xwei(t_and_workpackage):
    '''
    Function to calculate the xWEI for one timestep for whole Germany. One slice at a time is processed
    :param timestep: time index of the whole dataset of slice to be processed
    :param my_work_package:
    :return:
    '''

    #print(f"Starting worker {mp.current_process().pid} ")
    timestep, my_work_package, file_counter = t_and_workpackage
    max_duration = my_work_package.max_duration
    dgev_parms = my_work_package.dgev_params
    vtx = my_work_package.weights[0]
    wts = my_work_package.weights[1]

    t = timestep

    #@Saikat: Change variable name here accordingly
    chunk = my_work_package.input_array["Rainfall"].isel(
        time=slice(t - int(max_duration), t)).values

    date = my_work_package.input_array.time[t].values
    # to speed up the later calculations every duration gets aggregated now
    chunk = create_duration_array(chunk)

    #@Saikat: Again the durations that you have to change
    #@Saikat: It is crucial that you change the cell_size_km2 here according to your grid!
    xwei = dgev_xwei_for_array(chunk, dgev_parms, vtx, wts,
                                   duration_levels=['01', '03', '06', '30', '60', '90'],
                                   nan_tolerance={'01': 1, '03': 3, '06': 5, '30': 27, '60': 54, '90': 81},
                                   cell_size_km2=30.98)


    res = pd.DataFrame({"date": [date], "xwei": [round(xwei, 1)]})
    res.to_csv(f"{path}output/tracking/{location}/{file_counter}_xwei.csv", index=False)

    return res


input_list = list(np.arange(90, xr_all.sizes["time"]))
# the counter is for the later merging of the files
file_counter = np.arange(len(input_list))

work_package = WorkPackage(dgev_parms, xr_all, weights, max_duration=90,
                           max_counter=[-1])

input_list = [(i, work_package) for i in input_list]
input_list = tuple(zip(input_list, file_counter))
input_list = [(i[0][0], i[0][1], i[1]) for i in input_list]


res_list = []
for counter, i in enumerate(input_list):
    print(counter)
    xwei_df = calc_xwei(i)
    res_list.append(xwei_df)


xwei_timeseries = pd.concat(res_list)

if not os.path.exists(f"{path}output/xwei_timeseries"):
    os.mkdir(f"{path}output/xwei_timeseries")

#@Saikat: Change filename
xwei_timeseries.to_csv(f"{path}output/xwei_timeseries/xwei_2000_2024_{location}.csv", index=False)

