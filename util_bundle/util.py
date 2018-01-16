#!/usr/bin/env python3
import numpy as np


# File handling

def read_data_file(filename, max_rows=None):
    with open(filename, 'r') as f_in:
        first_line = f_in.readline().strip()
        first_field = first_line.split('\t')[0]

        try:  # no header
            float(first_field)
            npa_data = np.genfromtxt(filename, delimiter="\t", dtype='f8', max_rows=max_rows)
            l_labels = None
        except ValueError:  # with header
            npa_data = np.genfromtxt(filename, delimiter="\t", dtype='f8', names=True, max_rows=max_rows)

            f_in.seek(0)
            header = f_in.readline().strip()
            l_labels = header.split('\t')
            l_labels = [_label.strip() for _label in l_labels]

            # The number of data fields may exceed the number of headers, remove the last column
            if len(l_labels) < len(npa_data[0]):
                names = list(npa_data.dtype.names)[:len(l_labels)]
                npa_data = npa_data[names]

        return l_labels, npa_data


def save_data_file(filename, header, data):
    """
    This function is symmetric with read_data_file().
    """
    if header is not None:
        header_ = '\t'.join(header)
        np.savetxt(filename, data, fmt='%.5g', delimiter='\t', comments='', newline='\n', header=header_)
    else:
        np.savetxt(filename, data, fmt='%.5g', delimiter='\t', comments='', newline='\n')


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
    #     "/Users/fairly/Documents/workspace_cpp/weijian_origin/cmake-build-debug/bin/output/20170504-175200/result
    # .dat",
    #     truncate_to=2)
    pass


def write_scalar_vtk(volume, ratio_ztox, filename='3D_reconstruction', ifbinary=False):
    z, y, x = volume.shape

    filename += '.vtk'

    # a mapping from np.dtype to types that vtk supports
    m_vtk_dtype = {
        np.bool_: 'bit',
        np.uint8: 'unsigned_char',
        np.uint16: 'unsigned_short',
        np.float64: 'double',
    }
    vtk_dtype = m_vtk_dtype[volume.dtype.type]

    if ifbinary:
        with open(filename, 'w') as fid:
            # Write a header defined by vtk
            print('# vtk DataFile Version 3.0', file=fid)
            print('vtk output', file=fid)
            print('BINARY', file=fid)
            print('DATASET STRUCTURED_POINTS', file=fid)
            print('DIMENSIONS %d %d %d' % (x, y, z), file=fid)
            print('SPACING 1 1 %.1f' % ratio_ztox, file=fid)
            print('ORIGIN 0 0 0', file=fid)
            print('POINT_DATA  %d' % (x * y * z), file=fid)
            print('SCALARS ImageFile %s' % vtk_dtype, file=fid)
            print('LOOKUP_TABLE default', file=fid)
            print(file=fid)

        with open(filename, 'ab') as fid:
            if volume.dtype.type == np.bool_:
                volume = np.packbits(volume)

            # Write bytes of the volume
            # When the volume is too big, directly write will fail, break it down.
            fid.write(volume[0:int(np.math.floor(z / 2)), :, :].tobytes())
            fid.write(volume[int(np.math.floor(z / 2)):, :, :].tobytes())
    else:
        # Write ASCIIs of the volume
        with open(filename, 'w') as fid:
            # Write a header defined by vtk
            print('# vtk DataFile Version 3.0', file=fid)
            print('vtk output', file=fid)
            print('ASCII', file=fid)
            print('DATASET STRUCTURED_POINTS', file=fid)
            print('DIMENSIONS %d %d %d' % (x, y, z), file=fid)
            print('SPACING 1 1 %.1f' % ratio_ztox, file=fid)
            print('ORIGIN 0 0 0', file=fid)
            print('POINT_DATA  %d' % (x * y * z), file=fid)

            print('SCALARS ImageFile %s' % vtk_dtype, file=fid)
            print('LOOKUP_TABLE default', file=fid)
            print(file=fid)

            # write data
            if volume.dtype.type == np.bool_:
                print(*(np.reshape(volume.astype(np.uint8), x * y * z).tolist()), file=fid)
            else:
                print(*(np.reshape(volume, x * y * z).tolist()), file=fid)

    print(filename + ' SCALAR volume writing done!')
