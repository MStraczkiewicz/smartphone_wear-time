"""Functions to estimate and plot smartphone wear-time using Beiwe
accelerometer sensor.
"""

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys

from dateutil import tz
import numpy as np
import pandas as pd


def get_noise_level(input_path: str, output_path: str) -> None:
    """Estimate smartphone accelerometer noise level.

    Args:
        input_path : string
            local repository with beiwe folders (IDs) for a given study
        output_path : string
            local repository to store results
    """
    print("(1/3) Estimating noise levels.../n")
    # analysis parameters
    analysis_window = 7

    # define variables
    earth_acc = 9.80665

    # TZ info
    fmt = '%Y-%m-%d %H_%M_%S'

    # list available beiwe_ids
    beiwe_ids = os.listdir(input_path)

    # preallocate outcome
    df = []

    for k, beiwe_id in enumerate(beiwe_ids):
        sys.stdout.write(str(k) + " " + beiwe_id + "/n")

        # list beiwe files
        beiwe_files_list = os.listdir(os.path.join(input_path,
                                                   beiwe_id,
                                                   "accelerometer"))

        if len(beiwe_files_list) > 0:  # folder cannot be empty
            # transform all files in folder to datelike format
            if "+00_00.csv" in beiwe_files_list[0]:
                beiwe_files_dt = [datetime.strptime(
                    file.replace("+00_00.csv", ""), fmt)
                    for file in beiwe_files_list]
            else:
                beiwe_files_dt = [datetime.strptime(
                    file.replace(".csv", ""), fmt)
                    for file in beiwe_files_list]

            # list days of data collection
            beiwe_files_days = [t_i - timedelta(hours=t_i.hour)
                                for t_i in beiwe_files_dt]
            beiwe_unique_days = list(set(beiwe_files_days))
            beiwe_unique_days.sort()

            # trim the timeline - you don't need to process all days just yet
            if len(beiwe_unique_days) > analysis_window:
                beiwe_unique_days = beiwe_unique_days[0:analysis_window]

            # create a timeline
            timeline = np.array(beiwe_unique_days)
            timeline = timeline[0:analysis_window]

            noise_level = np.empty((len(beiwe_unique_days), 24))
            noise_level.fill(np.nan)

            # loop over available days
            for d, current_day in enumerate(beiwe_unique_days):
                # find files from a given day
                beiwe_files_ind = [np.where(np.isin(beiwe_files_days,
                                            current_day))[0].tolist()][0]
                beiwe_files_current_day = [beiwe_files_list[i]
                                           for i in beiwe_files_ind]
                # loop over files within day
                for f, file_name in enumerate(beiwe_files_current_day):
                    data = pd.read_csv(os.path.join(input_path,
                                                    beiwe_id,
                                                    "accelerometer",
                                                    file_name))

                    t = np.array(data["UTC time"])
                    h = datetime.strptime(t[0].replace("T", " "),
                                          '%Y-%m-%d %H:%M:%S.%f')
                    h = (h - timedelta(minutes=h.minute) -
                         timedelta(seconds=h.second)).hour

                    x = np.array(data["x"])
                    y = np.array(data["y"])
                    z = np.array(data["z"])

                    vm = np.sqrt(x**2+y**2+z**2)
                    # standardize measurement to gravity units (g)
                    if np.mean(vm) > 5:
                        x = x/earth_acc
                        y = y/earth_acc
                        z = z/earth_acc

                    # sum of std in each axis
                    act_lev = np.sum([np.std(x), np.std(y), np.std(z)])

                    noise_level[d, h] = act_lev

        # estimate noise level as 5 times the lowest observed noise level
        temp = noise_level.flatten()
        temp.sort()
        typical_noise_level = 5*temp[np.nonzero(temp)[0][0]]
        if typical_noise_level > 0.01:
            typical_noise_level = 0.01

        d_to_append = pd.DataFrame({"Beiwe_id": beiwe_id,
                                    "Noise_level":
                                        np.array(typical_noise_level)},
                                   index=[k])
        df.append(d_to_append)

    # create directory and save results
    os.makedirs(os.path.join(output_path, "noise_level"), exist_ok=True)

    df = pd.concat(df)
    df.to_csv(os.path.join(output_path, "noise_level", "noise_level.csv"))

    print("/n(1/3) Noise levels estimated./n")


def rle(inarray: np.ndarray) -> tuple:
    """Runs length encoding.

    Args:
        inarray: array of Boolean values
            input for run length encoding

    Returns:
        Tuple of running length, starting index, and running values
    """
    array_length = len(inarray)
    if array_length == 0:
        return (None, None, None)
    else:
        pairwise_unequal = inarray[1:] != inarray[:-1]  # pairwise unequal
        ind = np.append(np.where(pairwise_unequal),
                        array_length - 1)  # must include last element position
        run_length = np.diff(np.append(-1, ind))  # run lengths
        start_ind = np.cumsum(np.append(0, run_length))[:-1]  # position
        val = inarray[ind]  # running values
        return run_length, start_ind, val


def get_smartphone_weartime(input_path: str, output_path: str) -> None:
    """Estimate smartphone weartime.

    Args:
        input_path : string
            Path to local repository with beiwe folders (IDs) for a given study
        output_path : string
            Path to local repository to store results
    """
    print("(2/3) Estimating wear-time.../n")
    # define variables
    fs = 10  # desired sampling rate (in Hertz (Hz))
    earth_acc = 9.80665

    # other thresholds
    default_noise = 0.01

    # TZ info
    fmt = '%Y-%m-%d %H_%M_%S'
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('UTC')  # change to desired TZ

    # list available beiwe_ids
    beiwe_ids = os.listdir(input_path)

    # get smartphone accelerometer noise data
    noise_data = pd.read_csv(os.path.join(output_path,
                                          "noise_level", "noise_level.csv"))

    for k, beiwe_id in enumerate(beiwe_ids):
        sys.stdout.write(str(k) + " " + beiwe_id + "/n")

        # assign noise level threshold
        try:
            noise_data_ind = noise_data["Beiwe_id"].to_list().index(beiwe_id)
            noise_level = noise_data["Noise_level"][noise_data_ind]
        except ValueError:  # Noise level were never assigned
            noise_level = default_noise

        # list beiwe files
        beiwe_files_list = os.listdir(os.path.join(input_path,
                                                   beiwe_id,
                                                   "accelerometer"))

        if len(beiwe_files_list) > 0:  # folder cannot be empty
            # transform all files in folder to datelike format
            if "+00_00.csv" in beiwe_files_list[0]:
                beiwe_files_dt = [datetime.strptime(
                    file.replace("+00_00.csv", ""), fmt)
                    for file in beiwe_files_list]
            else:
                beiwe_files_dt = [datetime.strptime(
                    file.replace(".csv", ""), fmt)
                    for file in beiwe_files_list]

            # list days of data collection
            beiwe_files_days = [t_i - timedelta(hours=t_i.hour)
                                for t_i in beiwe_files_dt]
            beiwe_unique_days = list(set(beiwe_files_days))
            beiwe_unique_days.sort()

            # create a timeline
            timeline_days = np.array(beiwe_unique_days)
            timeline_mins = []

            # allocate memory to store outcome
            # 1) collected data in seconds per minute
            cd_secs = np.zeros((len(timeline_days), 24*60))
            # 2) active-time in seconds per minute
            at_secs = np.zeros((len(timeline_days), 24*60))

            # loop over available days
            for d, current_day in enumerate(timeline_days):
                # find files from a given day
                beiwe_files_ind = [np.where(np.isin(beiwe_files_days,
                                            current_day))[0].tolist()][0]
                beiwe_files_current_day = [beiwe_files_list[i]
                                           for i in beiwe_files_ind]
                # loop over files within day
                for f, file_name in enumerate(beiwe_files_current_day):
                    print(file_name)
                    # load data
                    data = pd.read_csv(os.path.join(input_path,
                                                    beiwe_id,
                                                    "accelerometer",
                                                    file_name))
                    t = np.array(data["UTC time"])
                    x = np.array(data["x"])
                    y = np.array(data["y"])
                    z = np.array(data["z"])

                    # convert t to datetime
                    t = [datetime.strptime(t_i.replace("T", " "),
                                           "%Y-%m-%d %H:%M:%S.%f")
                         for t_i in t]

                    # adjust t to new TZ
                    t = [t_i.replace(tzinfo=from_zone).astimezone(to_zone)
                         for t_i in t]

                    # create variables for bout-fiding and sample-time-matching
                    t_seconds = [t_i -
                                 timedelta(microseconds=t_i.microsecond)
                                 for t_i in t]

                    hour_start = t_seconds[0]
                    hour_start = (hour_start -
                                  timedelta(minutes=hour_start.minute) -
                                  timedelta(seconds=hour_start.second))
                    hour_end = hour_start + timedelta(hours=1)

                    # find seconds with enough samples
                    t_sec_bins = pd.date_range(hour_start,
                                               hour_end, freq='S').tolist()
                    samples_per_sec, t_sec_bins = np.histogram(t_seconds,
                                                               t_sec_bins)

                    # there is a case of leap hours (to be addressed)
                    if len(t_sec_bins) > 1:
                        # seconds with enough samples / should be == fs
                        samples_enough = samples_per_sec >= (fs - 1)

                        # find bouts with sufficient duration (e.g., 5s)
                        N, BI, B = rle(samples_enough)
                        bout_start = BI[B & (N >= 5)]
                        bout_duration = N[B & (N >= 5)]

                        # loop over bouts (on-cycle)
                        for b, b_datetime in enumerate(bout_start):
                            # create a list with second-level timestamps
                            bout_time = pd.date_range(
                                t_sec_bins[int(bout_start[b])],
                                t_sec_bins[int(bout_start[b]) +
                                           int(bout_duration[b])-1],
                                freq='S').tolist()

                            # transform strings to datetime
                            bout_time = [t_i.to_pydatetime()
                                         for t_i in bout_time]

                            # find observations in this bout
                            acc_ind = np.isin(t_seconds, bout_time)
                            x_bout = x[acc_ind]
                            y_bout = y[acc_ind]
                            z_bout = z[acc_ind]

                            # standardize measurement to gravity units (g)
                            vm_bout = np.sqrt(x_bout**2+y_bout**2+z_bout**2)
                            if np.mean(vm_bout) > 5:
                                x_bout = x_bout/earth_acc
                                y_bout = y_bout/earth_acc
                                z_bout = z_bout/earth_acc

                            # sum of std in each axis
                            act_lev = np.sum([np.std(x_bout),
                                              np.std(y_bout),
                                              np.std(z_bout)])

                            # calculate available data
                            ind = np.floor([(t_i -
                                             t_i.replace(
                                                 hour=0,
                                                 minute=0,
                                                 second=0,
                                                 microsecond=0)).seconds/60
                                            for t_i in bout_time])
                            for i, t_i in enumerate(ind):
                                cd_secs[d, int(t_i)] += 1

                            # proceed only if there was any activity
                            if act_lev > noise_level:
                                # add info on smartphone wear
                                for i, t_i in enumerate(ind):
                                    at_secs[d, int(t_i)] += 1

                # add minute-level timestamps for current_day
                next_day = current_day + timedelta(days=1)
                timeline_to_add = pd.date_range(current_day,
                                                next_day,
                                                freq='Min')[0:-1].tolist()
                timeline_mins += timeline_to_add

            # create directory and save results
            os.makedirs(os.path.join(output_path, "wear_time"), exist_ok=True)

            # estimate wear-time in 1-minute windows
            wt_min = at_secs
            wt_min[wt_min > 0] = 1  # if there was any activity a given min.

            d = {"time": np.array(timeline_mins),
                 "collected_data": cd_secs.flatten(),
                 "active_time": at_secs.flatten(),
                 "wear-time": wt_min.flatten()}
            df = pd.DataFrame(data=d)
            df.to_csv(os.path.join(output_path, "wear_time",
                                   beiwe_id + "_weartime.csv"))

    print("/n(2/3) Wear-time estimated./n")


def estimate_smartphone_weartime(input_path: str, output_path: str) -> None:
    """Estimate smartphone accelerometer noise level.

    Args:
        input_path : string
            local repository with beiwe folders (IDs) for a given study
        output_path : string
            local repository to store results
    """
    get_noise_level(input_path, output_path)
    get_smartphone_weartime(input_path, output_path)


# Change input_path to path with Beiwe IDs
input_path = ("C:/Users/mstra/Documents/data/STAR")

# Change output_path to your desired destination
output_path = ("C:/Users/mstra/Documents/Python/star")

estimate_smartphone_weartime(input_path, output_path)
