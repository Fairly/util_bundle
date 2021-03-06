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


def dt_recognition(data, time_prefix='t'):
    """
    The unit of dt is msec.

    :param data:
    :param time_prefix:
    :return: float
    """
    time_name = ''
    for header in data.dtype.names:
        if header.startswith(time_prefix):
            time_name = header
            break

    if time_name == '':
        raise DataError("Cannot extract dt, "
                        "please check 'time' is in the data file.")

    i_t = data.dtype.names.index(time_name)

    # dt equals the difference of t between 2 consecutive rows
    dt = data[1][i_t] - data[0][i_t]

    if dt > 2:
        print('Warning! dt > 2 ms.')

    return dt


def ap_recognition(data):
    """

    Return:
        [(start_point, BCL)...]

    Raise:
        DataError
    """
    i_t, i_v, i_dvdt, _, _ = find_fields(data)
    start_points = []  # of APs
    dt = dt_recognition(data)
    jump_time = 5  # [ms]
    jump_step = int(jump_time/dt)   # jump over a short time to avoid local maximum in rare cases

    if i_dvdt is None:
        data, i_dvdt = calculate_dvdt(data, dt, i_v)

    # find all stimuli
    if_find_upstroke = False
    i = 0
    while i < len(data):
        row = data[i]

        if if_find_upstroke is False and row[i_dvdt] > 5:
            # spot a upstroke
            if_find_upstroke = True
            start_points.append(row[i_t])
            i += jump_step

        elif if_find_upstroke is True and row[i_dvdt] < 0:
            if_find_upstroke = False

        i += 1

    # generate the return list
    l_ap = []
    for i, stim in enumerate(start_points):
        if i == len(start_points) - 1:
            l_ap.append((stim, data[-1][i_t] - stim))
        else:
            l_ap.append((stim, start_points[i + 1] - stim))

    return l_ap


def get_range(beat, dt, l_ap, time_offset=0):
    """
    Given the beat No., dt, and list of APs returned by ``AP_recognition``, return the
    start and end row number in data.

    :param beat:
    :param dt:
    :param l_ap:
    :return:
    """
    # line number of data file starts from 1, so minus 1 here
    if int((l_ap[beat][0]-time_offset) / dt) == 0:
        beat_start = 0
    else:
        beat_start = int((l_ap[beat][0]-time_offset) / dt) - 1
    beat_end = int((l_ap[beat][0]-time_offset) / dt + l_ap[beat][1] / dt) - 1
    return beat_start, beat_end


def measure(infilename, celltype='n', drop_last=False, tissue=False):
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
    dt = dt_recognition(data)
    i_t, i_v, i_dvdt, i_ca_i, i_na_i = find_fields(data)

    if i_t is None:
        print("Cannot find 'time' in the file. "
              "Are you processing a wrong file?")
        exit(1)
    if i_dvdt is None and i_v is None:
        print("Both 'dvdt' and 'V_m' are not in the file. Cannot calculate dV/dt, exit.")
        exit(1)

    if i_dvdt is None:
        data, i_dvdt = calculate_dvdt(data, dt, i_v)

    if celltype == 'n':
        l_ap = ap_recognition(data)
        if drop_last:
            l_ap = l_ap[:-1]

        num_bcl = len(l_ap)

        if num_bcl == 0:
            print("No APs detected in input.")
            exit(1)

        upstroke_duration = 8

        # Calculation starts here
        result_header_list = ["Beat", "Stim_t", "AP_Min", "AP_Max", "AP_Amp",
                              "dVdt_Max", "dVdt_Max_t",  "I_CaL_Max",  # maximum dVdt and the time of it
                              "Ca_i_Max", "Ca_i_Dia", "Ca_i_Amp",
                              "Na_i_Dia", "Ca_i_tau",
                              "APD20", "APD_25", "APD_30", "APD_50", "APD_70", "APD_75", "APD_80", "APD_90"]
        result = np.zeros((num_bcl, len(result_header_list)))  # Initialise results

        # Beats count
        for i in range(num_bcl):
            # For each beat
            result[i, 0] = i  # The beat written in file starts from 0

        tmp = []
        # First loop. Calculate characteristics except APDs
        for beat in range(0, num_bcl):

            beat_start, beat_end = get_range(beat, dt, l_ap, data[0][i_t])

            # In a single beat
            min_ap_beforepeak = +1000
            min_ap_afterpeak = +1000
            max_ap = -1000
            max_dvdt = -1000
            max_dvdt_t = -1000
            max_ICaL = 0
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
                if row['I_CaL'] and abs(row['I_CaL']) > abs(max_ICaL):
                    max_ICaL = row['I_CaL']

                if i_ca_i is not None:
                    if row[i_ca_i] > max_ca_i:
                        max_ca_i = row[i_ca_i]
                    if row[i_ca_i] < min_ca_i_afterpeak and row[i_t] >= data[beat_start][i_t] + upstroke_duration:
                        min_ca_i_afterpeak = row[i_ca_i]
                    if row[i_ca_i] < min_ca_i_beforepeak and row[i_t] < data[beat_start][i_t] + upstroke_duration:
                        min_ca_i_beforepeak = row[i_ca_i]
                if i_na_i is not None:
                    if row[i_na_i] < min_na_i:
                        min_na_i = row[i_na_i]

            amp_ap = max_ap - min_ap_beforepeak
            amp_ca_i = max_ca_i - min_ca_i_beforepeak

            if tissue:
                min_ca_afterandbefore = min_ca_i_beforepeak if min_ca_i_beforepeak < min_ca_i_afterpeak \
                    else min_ca_i_afterpeak
                amp_ca_i = max_ca_i - min_ca_afterandbefore
            else:
                min_ca_afterandbefore = min_ca_i_afterpeak

            tmp = [l_ap[beat][0], min_ap_afterpeak, max_ap, amp_ap, max_dvdt, max_dvdt_t, max_ICaL,
                   max_ca_i, min_ca_afterandbefore, amp_ca_i, min_na_i]
            result[beat, 1:len(tmp) + 1] = tmp
            # End of a single beat

        # Second loop. Calculate APDs
        for beat in range(0, num_bcl):

            beat_start, beat_end = get_range(beat, dt, l_ap, data[0][i_t])

            # In a single beat
            ap_90 = result[beat, 3] - 0.9 * (result[beat, 3] - result[beat, 2])
            ap_80 = result[beat, 3] - 0.8 * (result[beat, 3] - result[beat, 2])
            ap_75 = result[beat, 3] - 0.75 * (result[beat, 3] - result[beat, 2])
            ap_70 = result[beat, 3] - 0.7 * (result[beat, 3] - result[beat, 2])
            ap_50 = result[beat, 3] - 0.5 * (result[beat, 3] - result[beat, 2])
            ap_30 = result[beat, 3] - 0.3 * (result[beat, 3] - result[beat, 2])
            ap_25 = result[beat, 3] - 0.25 * (result[beat, 3] - result[beat, 2])
            ap_20 = result[beat, 3] - 0.20 * (result[beat, 3] - result[beat, 2])
            cai_tau = result[beat, 8] * (1 - 1 / math.e) + result[beat, 7] / math.e

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
                    apd_25 = row[i_t] - result[beat, 6]  # result[beat, 6] is the time of dVdt_max

                # APD20
                if row[i_v] <= ap_20 <= data[current_row_num - 1][i_v]:
                    apd_20 = row[i_t] - result[beat, 6]  # result[beat, 6] is the time of dVdt_max

                # APD30
                if row[i_v] <= ap_30 <= data[current_row_num - 1][i_v]:
                    apd_30 = row[i_t] - result[beat, 6]

                # APD50
                if row[i_v] <= ap_50 <= data[current_row_num - 1][i_v]:
                    apd_50 = row[i_t] - result[beat, 6]

                # APD70
                if row[i_v] <= ap_70 <= data[current_row_num - 1][i_v]:
                    apd_70 = row[i_t] - result[beat, 6]

                # APD75
                if row[i_v] <= ap_75 <= data[current_row_num - 1][i_v]:
                    apd_75 = row[i_t] - result[beat, 6]

                # APD80
                if row[i_v] <= ap_80 <= data[current_row_num - 1][i_v]:
                    apd_80 = row[i_t] - result[beat, 6]

                # APD90
                if row[i_v] <= ap_90 <= data[current_row_num - 1][i_v]:
                    apd_90 = row[i_t] - result[beat, 6]

                # Time constant of Cai decay
                if row[i_ca_i] <= cai_tau <= data[current_row_num - 1][i_ca_i]:
                    tau_decay_cai = row[i_t] - data[beat_start][i_t]

            # End of a single beat
            result[beat, len(tmp) + 1:] = tau_decay_cai, apd_20, apd_25, apd_30, apd_50, apd_70, apd_75, apd_80, apd_90
    elif celltype == 'p':
        i_t, i_v, i_dvdt, i_ca_i, i_na_i = find_fields(data)

        if i_t is None or i_v is None or i_dvdt is None:
            print("Data file is not in right format. "
                  "Please check there are 't', 'dvdt',and 'V' fields in the file.")
            exit(1)

        lowv_found = False
        highv_found = False
        min_potential = []
        max_potential = []
        top_slope = []
        tmin_potential = []
        dvdtmax = []
        apd_start = []
        vdvdtmax = []
        top = [0]
        ddr = [0]
        apd50 = []
        apd90 = []
        cycle_length = []

        # initialize
        dvdtold = 100000
        Vold = 100000

        param_counter = 0
        for row in data:
            dvdt = row[i_dvdt]

            if not lowv_found and dvdt >= 0.0 >= dvdtold:
                lowv_found = True

                min_potential.append(Vold)
                tmin_potential.append(row[i_t])

                # resize
                dvdtmax.append(0)
                apd_start.append(0)
                vdvdtmax.append(0)

            if lowv_found and dvdtold >= 0.0 >= dvdt:
                highv_found = True

                max_potential.append(Vold)
                top_slope.append((max_potential[param_counter] - min_potential[param_counter]) / (row[i_t] - tmin_potential[param_counter]))

            if lowv_found and dvdt > dvdtmax[param_counter]:
                dvdtmax[param_counter]   = dvdt
                apd_start[param_counter] = row[i_t]
                vdvdtmax[param_counter]  = row[i_v]

            if param_counter > 0 and lowv_found and dvdtold <= top_slope[param_counter - 1] <= dvdt:
                top.append(Vold)
                ddr.append((Vold - min_potential[param_counter]) / (row[i_t] - tmin_potential[param_counter]))

            if highv_found and row[i_v] <= 0.5 * max_potential[param_counter] + 0.5 * min_potential[
                param_counter] <= Vold:
                apd50.append(row[i_t] - apd_start[param_counter])

            if highv_found and row[i_v] <= 0.1 * max_potential[param_counter] + 0.9 * min_potential[
                param_counter] <= Vold:
                apd90.append(row[i_t] - apd_start[param_counter])

                if param_counter == 0:
                    cycle_length.append(0)
                elif param_counter > 0:
                    cycle_length.append(apd_start[param_counter] - apd_start[param_counter - 1])

                highv_found = False
                lowv_found  = False

                param_counter += 1

            dvdtold = dvdt
            Vold = row[i_v]

        result_header_list = ["MDP", "OS", "dvdt_max", "APD50", "APD90", "CL", "DDR", "TOP"]
        result = [list(r) for r in zip(min_potential, max_potential, dvdtmax, apd50, apd90, cycle_length, ddr, top)]

    else:
        raise Exception("Unknown cell type.")

    # All calculation ends
    # print result

    # Output results
    infilepath, _infilename = os.path.split(infilename)
    out_file_name = 'measurement_' + _infilename
    out_file_name = os.path.join(infilepath, out_file_name)

    # Construct headers
    result_header = '\t'.join(result_header_list)
    # Save
    np.savetxt(out_file_name, result, fmt="%.4f", delimiter='\t', header=result_header, comments='')

    print("measurement of %s is done." % infilename)


def calculate_dvdt(data, dt, i_v):
    print("Warning: no 'dvdt' in the file, calculated by time and V, maybe not accurate.")
    if i_v is None:
        print('Error: No V_m in the result file. Cannot calculate dvdt.')
        exit(1)

    array_dvdt = np.zeros(len(data))
    old_V = data[0][i_v]
    new_V = data[1][i_v]
    dvdt = 0
    for i, row in enumerate(data):
        if i + 1 > len(data):
            pass
        else:
            dvdt = (new_V - old_V) / dt
            old_V = new_V
            if i + 2 < len(data):
                new_V = data[i + 2][i_v]
        array_dvdt[i] = dvdt
    # add a column of 'dvdt' to the data
    from numpy.lib.recfunctions import append_fields
    data = append_fields(data, 'dvdt', array_dvdt)
    i_dvdt = data.dtype.names.index('dvdt')
    return data, i_dvdt


def find_fields(data):
    # Find data fields
    time_name = None
    voltage_name = None
    dvdt_name = None
    cai_name = None
    na_myo_name = None

    for header in data.dtype.names:
        if header.lower() == 't' or header.lower() == 'time':
            time_name = header
        elif 'v_m' == header.lower() or header.lower() == 'v' or 'ap' == header.lower():
            voltage_name = header
        elif 'dvdt' in header.lower():
            dvdt_name = header
        elif header.startswith('Ca_i'):
            cai_name = header
        elif header.startswith('Na_myo') or header.startswith('Na_i'):
            na_myo_name = header

    # Record some indices
    i_t = data.dtype.names.index(time_name) if time_name is not None else None
    i_v = data.dtype.names.index(voltage_name) if voltage_name is not None else None
    i_dvdt = data.dtype.names.index(dvdt_name) if dvdt_name is not None else None
    i_ca_i = data.dtype.names.index(cai_name) if cai_name is not None else None
    i_na_i = data.dtype.names.index(na_myo_name) if na_myo_name is not None else None
    return i_t, i_v, i_dvdt, i_ca_i, i_na_i


if __name__ == '__main__':
    filename = sys.argv[1]
    measure(filename)
