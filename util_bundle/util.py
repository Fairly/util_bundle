#!/usr/bin/env python3
import shutil

import numpy as np

from util_bundle.measurement import ap_recognition, get_range


# File handling

def read_data_file(filename):
    npa_data = np.genfromtxt(filename, delimiter="\t", dtype='f8', names=True)
    with open(filename, 'r') as f_in:
        header = f_in.readline().strip()
        l_labels = header.split('\t')
    return l_labels, npa_data


def save_data_file(filename, header, data):
    """
    This function is symmetric with read_data_file().
    """
    header_ = '\t'.join(header)
    np.savetxt(filename, data, fmt='%f', delimiter='\t',
               comments='', newline='\n', header=header_)


def clean_result_for_plot(filename, add_underline=True, truncate_to=None,
                          reset_start_time=True, tail=True):  # TODO parameter `tail`
    """
    When plotting a result, it's common to reduce the size of the result file first.

    :param filename:
    :param add_underline: whether add a '_' mark to every field in header names
    :param reset_start_time: if the start time is not 0, whether to reset it to 0
    :param truncate_to: whether truncate the result file to a single beat (the final beat)
    :return:
    """
    backup = 'backup.dat'

    shutil.copyfile(filename, backup)

    headers, data = read_data_file(backup)

    if add_underline and not headers[0].endswith('_'):
        for i, tag in enumerate(headers):
            headers[i] = tag + '_'

    if truncate_to is not None:
        dt, beats = ap_recognition(data)
        start, _ = get_range(len(beats) - truncate_to, dt, beats)
        _, end = get_range(len(beats) - 1, dt, beats)
        data = data[start:end+1]

    if reset_start_time:
        time_offset = data[0][0] - (data[1][0] - data[0][0])  # time of 1st row minus dt
        for row in data:
            row[0] = row[0] - time_offset

    save_data_file(filename, headers, data)


# Plot utils

def normalize(_list, divisor=None):
    if divisor is None:
        divisor = float("-inf")
        for i in _list:
            if i > divisor:
                divisor = i

    return [i / divisor for i in _list]


def bar_arrange(num_groups, bars_in_group=1):
    """
    Return the x position of all groups of bars.

    :param num_groups: must be integer
    :param bars_in_group: must be integer
    :return:
    """
    basic_dis = 1
    _dis_between_bars = basic_dis
    _bar_width = 3 * basic_dis
    _width_of_group = _bar_width * bars_in_group + (bars_in_group - 1) * _dis_between_bars
    _dis_between_groups = _width_of_group / 2.0
    _dis_from_border = _width_of_group

    basic_dis = 1.0 / (_dis_from_border * 2
                       + _width_of_group * num_groups
                       + (num_groups - 1) * _dis_between_groups)

    _dis_between_bars = basic_dis
    _bar_width = 3 * basic_dis
    _width_of_group = _bar_width * bars_in_group + (bars_in_group - 1) * _dis_between_bars
    _dis_between_groups = _width_of_group / 2.0
    _dis_from_border = _width_of_group

    x_positions = []
    for i in range(bars_in_group):
        if i == 0:
            x_pos = [_dis_from_border + _bar_width / 2]
            for j in range(num_groups - 1):
                x_pos.append(x_pos[j] + _width_of_group + _dis_between_groups)
            x_positions.append(x_pos)
        else:
            x_pos = [x + _dis_between_bars + _bar_width for x in x_positions[i - 1]]
            x_positions.append(x_pos)

    return x_positions, _bar_width


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))  # outward by 10 points
            # spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


if __name__ == "__main__":
    print(bar_arrange(5, 3))

    # import matplotlib.pyplot as plt
    # x_pos, width = bar_arrange(2)
    # plt.bar(x_pos, [1, 2], align='center', width=width)
    # plt.show()

    # clean_result_for_plot(
    #     "/Users/fairly/Documents/workspace_cpp/weijian_origin/cmake-build-debug/bin/output/20170504-175200/result.dat",
    #     truncate_to=2)
    pass
