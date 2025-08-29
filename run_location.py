'''
Paul Voit Aug 29
This is basically a wrapper script that is supposed to automatize the whole
'''

import subprocess
import os
import pandas as pd
import datetime as dt

os.chdir("/home/voit/Radolan/skripte/maria_landslides")
#location = "LakeKivu_Congo"
#location = "MtHanang_Tanzania"
locations = pd.read_csv("/home/voit/Radolan/maria_landslides/Event_dates.csv")
location_list = locations.Region.to_list()
#location_list = ["BRA_11"]

# Function to remove the first number and underscore
# location_list = [i.split("_")[0] for i in locations]

for location in location_list:

    print(f'{dt.datetime.now()}: {location.upper()}')
    #calculate rollsums and yearmax
    print(f'{dt.datetime.now()}: calculating rolling sums')
    subprocess.run(['python', 'xr_rollsum.py', location], text=True)
    #fit dGEV with R
    print(f'{dt.datetime.now()}: fitting dGEV')
    subprocess.run(['Rscript', 'dgev_fit.R', location], text=True)
    # make xwei timeseries
    print(f'{dt.datetime.now()}: xWEI timeseries')
    subprocess.run(['python', 'xwei_timeseries_for_location.py', location], text=True, stdout=subprocess.DEVNULL)
    #make plots
    subprocess.run(['python', 'plot_xwei_series.py', location], text=True)
    #wei analysis of 90 days prior to the event
    print(f'{dt.datetime.now()}: wei analysis')
    subprocess.run(['python', 'wei_analysis.py', location], text=True)
    print(f'################{dt.datetime.now()}: FINISHED##############')


#which files are missing
# import glob
# there = glob.glob("/home/voit/Radolan/maria_landslides/output/plots/xwei*")
# there = [i.split("/")[-1] for i in there]
# there = [i.strip("xwei_") for i in there]
# there = [i.strip(".png") for i in there]
#
# missing = set(location_list) - set(there)