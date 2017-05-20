#!/usr/bin/env python3

import sys
import os
import math

import numpy as np


class DataError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def ap_recognition(data, time_prefix='t', stim_prefix='I_Stim'):
    """
    The unit of dt is msec.

    Return:
        (dt, [(start_point, BCL)...])

    Raise:
        DataError
    """
    time_name = ''
    stim_name = ''
    for header in data.dtype.names:
        if header.startswith(time_prefix):
            time_name = header
        elif header.startswith(stim_prefix):
            stim_name = header
    if time_name == '' or stim_name == '':
        raise DataError("Cannot extract BCL or dt "
                        "if time or stimulation is not in the data file.")

    i_t = data.dtype.names.index(time_name)
    i_i_stim = data.dtype.names.index(stim_name)

    # dt equals the difference of t between 2 consecutive rows
    dt = data[1][i_t] - data[0][i_t]

    # find all stimuli
    l_stim = []
    for i, row in enumerate(data):
        if row[i_i_stim] != 0 and (i == 0 or data[i - 1][i_i_stim] == 0):
            # spot a stimulus
            if i == len(data) - 1:
                # if there is something wrong in applying stimuli in the cell model,
                # an extra stimulus may appear at the end of data.
                # ignore this.
                continue

            if row[i_t] == 1:
                # although the starting time of a model running is 1,
                # the stimulus should be from 0
                l_stim.append(0.0)
            else:
                l_stim.append(row[i_t])

    l_ap = []
    for i, stim in enumerate(l_stim):
        if i == len(l_stim) - 1:
            l_ap.append((stim, len(data) - stim))
        else:
            l_ap.append((stim, l_stim[i + 1] - stim))

    return dt, l_ap


def get_range(beat, dt, l_ap):
    """
    Given the beat No., dt, and list of APs returned by ``AP_recognition``, return the
    start and end row number in data.

    :param beat:
    :param dt:
    :param l_ap:
    :return:
    """
    # line number of data file starts from 1, so minus 1 here
    if int(l_ap[beat][0] / dt) == 0:
        beat_start = 0
    else:
        beat_start = int(l_ap[beat][0] / dt) - 1
    beat_end = int(l_ap[beat][0] / dt + l_ap[beat][1] / dt) - 1
    return beat_start, beat_end


def measure(infilename):
    """
    Intend to calculate several values indicating AP characteristics.

    Beat, Min_AP, Max_AP, Amplitude_AP, dVdt_max, APD25, 30, 50, 70, 75, 80, 90.

    APDs are calculated from the time of maximum dVdt.

    Args:
        infilename: The input filename. It should have a header line. All data
                    fields are separated by spaces or tabs.
    """
    # read and save data to truncate them
    # noinspection PyTypeChecker
    data = np.genfromtxt(infilename, names=True, delimiter="\t")
    dt, l_ap = ap_recognition(data)
    num_bcl = len(l_ap)

    if num_bcl == 0:
        raise DataError("No APs detected in input.")

    # plot.save_data_file(infilename, header, data[int(-5*1000/dt):])

    # header, data = plot.read_data_file(infilename)

    upstroke_duration = 8

    # Calculation starts here
    result_header_list = ["Beat", "AP_Min", "AP_Max", "AP_Amp",
                          "dVdt_Max", "dVdt_Max_t",  # maximum dVdt and the time of it
                          "Ca_i_Max", "Ca_i_Dia", "Ca_i_Amp",
                          "Na_i_Dia", "Ca_i_tau",
                          "APD20", "APD_25", "APD_30", "APD_50", "APD_70", "APD_75", "APD_80", "APD_90"]
    result = np.zeros((num_bcl, len(result_header_list)))  # Initialise results

    # Beats count
    for i in range(num_bcl):
        # For each beat
        result[i, 0] = i  # The beat written in file starts from 0

    # Find data fields
    time_name = ''
    voltage_name = ''
    dvdt_name = ''
    cai_name = ''
    na_myo_name = ''
    for header in data.dtype.names:
        if header.startswith('t'):
            time_name = header
        elif header.startswith('V_m'):
            voltage_name = header
        elif header.startswith('d_dVdt'):
            dvdt_name = header
        elif header.startswith('Ca_i'):
            cai_name = header
        elif header.startswith('Na_myo'):
            na_myo_name = header
    if time_name == '' or voltage_name == '' or dvdt_name == '' \
            or cai_name == '' or na_myo_name == '':
        raise DataError("Cannot extract data fields from file: " + infilename)

    # Record some indices
    i_t = data.dtype.names.index(time_name)
    i_v = data.dtype.names.index(voltage_name)
    i_dvdt = data.dtype.names.index(dvdt_name)
    i_ca_i = data.dtype.names.index(cai_name)
    i_na_i = data.dtype.names.index(na_myo_name)

    tmp = []
    # First loop. Calculate characteristics except APDs
    for beat in range(0, num_bcl):

        beat_start, beat_end = get_range(beat, dt, l_ap)

        # In a single beat
        min_ap_beforepeak = +1000
        min_ap_afterpeak = +1000
        max_ap = -1000
        max_dvdt = -1000
        max_dvdt_t = -1000
        max_ca_i = -1000
        min_ca_i_beforepeak = 1000
        min_ca_i_afterpeak = 1000
        min_na_i = 1000

        for row in data[beat_start: beat_end]:
            # for some minimum values, there may be local minimums before or
            # after the AP peak, so record them separately
            if row[i_v] < min_ap_beforepeak and row[i_t] < data[beat_start][i_t] + upstroke_duration:
                min_ap_beforepeak = row[i_v]
            if row[i_v] < min_ap_afterpeak and row[i_t] >= data[beat_start][i_t] + upstroke_duration:
                min_ap_afterpeak = row[i_v]
            if row[i_v] > max_ap:
                max_ap = row[i_v]
            if row[i_dvdt] > max_dvdt:
                max_dvdt = row[i_dvdt]
                max_dvdt_t = row[i_t]
            if row[i_ca_i] > max_ca_i:
                max_ca_i = row[i_ca_i]
            if row[i_ca_i] < min_ca_i_afterpeak and row[i_t] >= data[beat_start][i_t] + upstroke_duration:
                min_ca_i_afterpeak = row[i_ca_i]
            if row[i_ca_i] < min_ca_i_beforepeak and row[i_t] < data[beat_start][i_t] + upstroke_duration:
                min_ca_i_beforepeak = row[i_ca_i]
            if row[i_na_i] < min_na_i:
                min_na_i = row[i_na_i]

        amp_ap = max_ap - min_ap_beforepeak
        amp_ca_i = max_ca_i - min_ca_i_beforepeak

        tmp = [min_ap_afterpeak, max_ap, amp_ap, max_dvdt, max_dvdt_t,
               max_ca_i, min_ca_i_afterpeak, amp_ca_i, min_na_i]
        result[beat, 1:len(tmp) + 1] = tmp
        # End of a single beat

    # Second loop. Calculate APDs
    for beat in range(0, num_bcl):

        beat_start, beat_end = get_range(beat, dt, l_ap)

        # In a single beat
        ap_90 = result[beat, 2] - 0.9 * (result[beat, 2] - result[beat, 1])
        ap_80 = result[beat, 2] - 0.8 * (result[beat, 2] - result[beat, 1])
        ap_75 = result[beat, 2] - 0.75 * (result[beat, 2] - result[beat, 1])
        ap_70 = result[beat, 2] - 0.7 * (result[beat, 2] - result[beat, 1])
        ap_50 = result[beat, 2] - 0.5 * (result[beat, 2] - result[beat, 1])
        ap_30 = result[beat, 2] - 0.3 * (result[beat, 2] - result[beat, 1])
        ap_25 = result[beat, 2] - 0.25 * (result[beat, 2] - result[beat, 1])
        ap_20 = result[beat, 2] - 0.20 * (result[beat, 2] - result[beat, 1])
        cai_tau = result[beat, 7] / math.e + result[beat, 6]

        # values below are not always foundable. If cannot find, assign 0.
        tau_decay_cai = 0
        apd_20 = 0
        apd_25 = 0
        apd_30 = 0
        apd_50 = 0
        apd_70 = 0
        apd_75 = 0
        apd_80 = 0
        apd_90 = 0

        for n, row in enumerate(data[beat_start: beat_end]):
            current_row_num = beat_start + n

            # APD25
            if row[i_v] <= ap_25 <= data[current_row_num - 1][i_v]:
                apd_25 = row[i_t] - result[beat, 5]  # result[beat, 5] is the time of dVdt_max

            # APD20
            if row[i_v] <= ap_20 <= data[current_row_num - 1][i_v]:
                apd_20 = row[i_t] - result[beat, 5]  # result[beat, 5] is the time of dVdt_max

            # APD30
            if row[i_v] <= ap_30 <= data[current_row_num - 1][i_v]:
                apd_30 = row[i_t] - result[beat, 5]

            # APD50
            if row[i_v] <= ap_50 <= data[current_row_num - 1][i_v]:
                apd_50 = row[i_t] - result[beat, 5]

            # APD70
            if row[i_v] <= ap_70 <= data[current_row_num - 1][i_v]:
                apd_70 = row[i_t] - result[beat, 5]

            # APD75
            if row[i_v] <= ap_75 <= data[current_row_num - 1][i_v]:
                apd_75 = row[i_t] - result[beat, 5]

            # APD80
            if row[i_v] <= ap_80 <= data[current_row_num - 1][i_v]:
                apd_80 = row[i_t] - result[beat, 5]

            # APD90
            if row[i_v] <= ap_90 <= data[current_row_num - 1][i_v]:
                apd_90 = row[i_t] - result[beat, 5]

            # Time constant of Cai decay
            if row[i_ca_i] <= cai_tau <= data[current_row_num - 1][i_ca_i]:
                tau_decay_cai = row[i_t] - data[beat_start][i_t]

        # End of a single beat
        result[beat, len(tmp) + 1:] = tau_decay_cai, apd_20, apd_25, apd_30, apd_50, apd_70, apd_75, apd_80, apd_90

    # All calculation ends
    # print result

    # Output results
    if '.' in infilename:
        suffix = infilename.split('.')[1]
        out_file_name = infilename.replace('.'+suffix, '_measurement.dat')
    else:
        out_file_name = infilename + '_measurement.dat'

    out_file_name = os.path.join(os.getcwd(), out_file_name)
    # Construct headers
    result_header = '\t'.join(result_header_list)
    # Save
    np.savetxt(out_file_name, result, fmt="%.4f",
               delimiter='\t', header=result_header, comments='')
    #

    print("measurement of %s is done." % infilename)


if __name__ == '__main__':
    filename = sys.argv[1]
    measure(filename)
