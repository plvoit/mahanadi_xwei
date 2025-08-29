'''
Paul Voit Aug 29 2025
Contains the functions to compute the xWEI and WEI
'''

import numpy as np
import pandas as pd
from scipy.stats import genextreme as gev_dist

def dgev_xwei_for_array(array, dgev_parms_dict, vtx, wts, max_rp=1000, resolution=1000,
              nan_tolerance={'01': 1, '02': 2, '04': 4, '06': 6, '12': 11, '24': 22, '48': 36, '72': 60},
              duration_levels=['01', '02', '04', '06', '12', '24', '48', '72'], cell_size_km2=1, return_eta_df=False
              ):
    '''
    This function calculates the xWEI for one timestep and cell in the RADKLIM dataset,
    The default is 72h and 3x3km. This means that around every cell a subset is taken with cell as center.
    For this subset the xWEI gets calculated. To speed up the process pre-calculated weights and vertices are used
    for the interpolation of the xWEI surface.

    :param array: numpy or xarray, data of precipitation
    :param dgev_parms: A dictionary of the dGEV parms (2D np.arrays). shape, mod_loc, scale_0, duration_offset
    and duration_exp

    :param vtx: np.array (type Int) precalculated vertices for interpolation
    :param wts: np.array (type Int) precalculated weights for interpolation
    :param max_rp: Int maximum considered return period. Usually set to 1000 (years).
    :param nan_tolerance: Dictionary
            The higher the duration, the bigger the moving window. By default if the moving window just contains one4
             NaN, the resulting moving window sum is als NaN. With this dictionary the tolerance towards NaN can be
             adjusted. The keys need to be the durations, the integer describes how many minimum non NaN-values need
             to be inside the window to still calculate the sum and thereby ignoring the NaN. This dictionary sets the
             values for xarray.rolling(...min_periods= DICCTIONARY VALUE).sum().
             If set to None, there will be no tolerance to NaN.
    :param duration_levels: list of durations in string format
    :return: float xWEI for the input array
    '''

    df = pd.DataFrame(np.ones((array.shape[1] * array.shape[2], len(duration_levels))), columns=duration_levels)
    array_copy = array.copy()  # this needs to be done because otherwise we cant write in the array

    for i in range(array_copy.shape[0]):

        duration = int(duration_levels[i])
        # dirty fix to fix nan
        array_copy[np.isnan(array_copy)] = 0

        # the dgev works with intensities rather then with rainfall amounts. Thats why we need to divide by duration
        Pu = gev_dist.cdf(array_copy[i]/duration, c=dgev_parms_dict[f"shape"],
                          loc=dgev_parms_dict[f"mu_{duration}"], scale=dgev_parms_dict[f"sigma_{duration}"])

        Pu[Pu == 1] = 0.999  # To fix the runtime error when division is by zero

        rp_array = 1 / (1 - Pu)

        result = _calc_eta(max_rp, rp_array, cell_size_km2)

        df[duration_levels[i]] = result

    eta = np.array(df).flatten("F")
    # interpolate values on finer grid using precalculated weights and vertices
    grid_z = interpolate(eta, vtx, wts)

    ########################
    # Just for plotting
    # eta_vals = df
    # resolution = 1000
    # precision = 1
    #
    # aggregated = 1
    #
    # coords = np.ones((len(df.columns) * len(df), 3))
    #
    # x_coords = list(np.arange(1, len(eta_vals) + 1, 1) * aggregated - (aggregated - 1)) * len(eta_vals.columns)
    #
    # durations = sorted([int(eta_vals.columns[i]) for i in range(len(eta_vals.columns))])
    # # from https://stackoverflow.com/questions/2449077/duplicate-each-member-in-a-list
    # y_coords = [val for val in durations for _ in (range(1, len(eta_vals) + 1))]
    #
    # # for sure this can be done better with pd.Dataframe.stack oder .melt(). Double list comprehension...
    # stacked_cols = [eta_vals[col] for col in eta_vals.columns]
    # z_coords = [item for sublist in stacked_cols for item in sublist]
    #
    # coords[:, 0] = np.log(x_coords)
    # coords[:, 1] = np.log(y_coords)
    # coords[:, 2] = z_coords
    #
    # ### interpolate
    # x_range = np.linspace(0, np.log(len(eta_vals) * aggregated), len(eta_vals))  # * aggregated
    # y_range = np.linspace(0, np.log(durations[-1]), resolution)
    # grid_x, grid_y = np.meshgrid(x_range, y_range)
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(grid_x, grid_y, grid_z.reshape(grid_x.shape[0], grid_y.shape[1]), cmap="viridis_r", edgecolor='none')
    # ax.set_title(f"rainfall: {array_copy[0, 1, 1]}")
    # ax.set_xlabel("Area")
    # ax.set_ylabel("Durations")
    # plt.show()

    ###############

    dx = np.log(len(df)) / len(df)
    dy = np.log(int(duration_levels[-1])) / resolution
    xwei = np.nansum(dx * dy * grid_z)

    if return_eta_df:
        return xwei, df

    else:
        return xwei



def interpolate(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret


def _calc_eta(max_rp, rp_array, cell_size_km2=1):
    """
    This function calculates the eta series for every timestep within one duration
    :param max_rp: Int. maximum possible return period
    :param rp_array: array with all the return periods
    :return: a list with all Eta-series for every timestep in one duration
    """

    '''
    correct all values exceeding max_return period threshold
    this might conflict with the (dirty) fix below
    '''

    if max_rp is not None:
        rp_array[rp_array > max_rp] = max_rp

    if len(rp_array.shape) > 2:
        # if the array has more than one timestep WEI needs to be calculated for each of them
        # and the one with highest maximum will be chosen in the end (?)
        result = list()

        for j in range(0, (rp_array.shape[0])):
            data = rp_array[j, :, :].flatten(order="c")

            '''
            set nan to zero. check paper S.149
            does this cause the floating point error/ runtime warning in get_log_Gta?
            wouldn't it be better to set it to 1? does tis have an influence on the WEI?
            Before nan was set to 0 causing runtime issues
            '''
            data[np.isnan(data)] = 1
            '''
            this is a dirty fix:
            sometimes the GEV fit doesn't seem to work properly:
            cells that have less precipitation than the maximum cell value observed
            end up with a return period of infinite because the cdf of the gev distribution
            resulted in zero. Of course it seems highly unlikely, especially in the case,
            where it is neighbouring cells. This happens for example for Event 11 on the duration 06h.
            Because the inf values mess with the following computation, all inf values will be set to the
            maximum.To keep track of this manipulation, a log file should be written.'
            21.12.21: This fix is obsolete now due to the infinte values 
            which get set to the self.max_rp threshold except when self.max_rp == None
            '''
            if max_rp is None:
                data[np.where(np.isinf(data))] = np.nanmax(data[data != np.inf])

            data = np.sort(data)[::-1]

            eta = _get_eta(data, cell_size_km2)

            result.append(eta.round(3))

    else:
        # if rp_array 2D then this
        data = rp_array[:, :].flatten(order="c")

        # set nan to zero. check paper S.149
        data[np.isnan(data)] = 1

        '''
        this is a dirty fix as above
        '''
        data[np.where(np.isinf(data))] = np.nanmax(data[data != np.inf])

        data = np.sort(data)[::-1]

        eta = _get_eta(data, cell_size_km2)

        result = eta.round(3)

    return result


def _get_eta(data: np.array, cell_size_km2=1):
    """
    Calculation of Eta according to Müller and Kaspar (2014)
    :param data: flattened array of return periods
    :return: np.array of Eta values
    """
    cumulated = np.cumsum(np.log(data))
    nr_cells = np.array([*range(len(data))]) + 1
    log_gta = cumulated / nr_cells
    area = nr_cells * cell_size_km2 #changed this to include different cell sizes
    r = np.sqrt(area) / np.sqrt(np.pi)
    eta = log_gta * r

    return eta


def get_max_eta(results_dict: dict):
    """
    Chooses the Eta series with highest maximum value for each duration.

    Parameters
    ----------
    results_dict : Dictionary
        Eta series returned from function get_WEI()

    Returns
    -------
    max_vals_dict : pd.Dataframe
        highest Eta series for each duration
    """

    max_vals = dict.fromkeys(results_dict.keys(), None)
    # areas_dict_new = dict.fromkeys(results_dict.keys(), None)

    for key in results_dict.keys():
        for i in range(0, len(results_dict[key])):
            if i == 0:
                max_vals[key] = results_dict[key][i]
            #  areas_dict_new[key] = area_dict[key]
            else:
                if np.nanmax(results_dict[key][i]) > np.nanmax(max_vals[key]):
                    max_vals[key] = results_dict[key][i]
            #     areas_dict_new[key] = area_dict[key][i]

    df = pd.DataFrame.from_dict(max_vals)
    check_max_eta(df)
    return pd.DataFrame.from_dict(max_vals).round(2)



def check_max_eta(eta_vals):
    for col in eta_vals.columns:
        if eta_vals[col].sum() == 0:
            print(f"WARNING: No Eta values for duration {col}h. xWEI result uncertain.")
