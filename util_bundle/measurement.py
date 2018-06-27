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

    return dt


def ap_recognition(data, time_prefix='t', stim_prefix='I_Stim'):
    """

    Return:
        [(start_point, BCL)...]

    Raise:
        DataError
    """
    # TODO Only support non-pacemaking cells
    time_name = ''
    dvdt_name = ''
    for header in data.dtype.names:
        if header.startswith(time_prefix):
            time_name = header
        elif 'dvdt' in header.lower():
            dvdt_name = header

    if time_name == '' or dvdt_name == '':
        raise DataError("Cannot find 'time' and 'dvdt' in the file. "
                        "Are you using a wrong structure in the file?")

    i_t = data.dtype.names.index(time_name)
    i_dvdt = data.dtype.names.index(dvdt_name)

    start_points = []  # of APs

    # find all stimuli
    if_find_upstroke = False
    for i, row in enumerate(data):
        if if_find_upstroke is False and row[i_dvdt] > 5:
            # spot a upstroke
            if_find_upstroke = True
            start_points.append(row[i_t])

        elif if_find_upstroke is True and row[i_dvdt] < 0:
            if_find_upstroke = False

    # generate the return list
    l_ap = []
    for i, stim in enumerate(start_points):
        if i == len(start_points) - 1:
            l_ap.append((stim, data[-1][i_t] - stim))
        else:
            l_ap.append((stim, start_points[i + 1] - stim))

    return l_ap


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


def measure(infilename, celltype='n'):
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

    if celltype == 'n':
        l_ap = ap_recognition(data)
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

        i_t, i_v, i_dvdt, i_ca_i, i_na_i = find_fields(data, infilename)

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
    elif celltype == 'p':
        i_t, i_v, i_dvdt, i_ca_i, i_na_i = find_fields(data, infilename)

        if i_t is None or i_v is None or i_dvdt is None:
            raise DataError("Data file is not in right format. "
                            "Please check there are 't', 'dvdt',and 'V' fields in the file.")

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
    if '.' in infilename:
        suffix = infilename.split('.')[1]
        out_file_name = infilename.replace('.' + suffix, '_measurement.dat')
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


def find_fields(data, infilename):
    # Find data fields
    time_name = None
    voltage_name = None
    dvdt_name = None
    cai_name = None
    na_myo_name = None

    for header in data.dtype.names:
        if header.startswith('t'):
            time_name = header
        elif header.startswith('V_m') or header == 'V':
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
