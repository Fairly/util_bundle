"""
Voltage_Clamp:
  DT: 0.1

  Target: I_to
  Protocol:
    HOLDING_V: -75
    HOLDING_TIME: 200

    # pre-pulse means a period between holding and test
    PREPULSE_V: -60
    PREPULSE_TIME: 150

    # test potential
    VC_START: -105.0
    VC_END: 35.01
    VC_STEP: 10.0
    DURATION: 200 # test_time

    POSTPULSE_V: 35
    POSTPULSE_TIME: 200

    TAIL_V: -75
    TAIL_TIME: 150
"""
import numpy as np
import matplotlib.pyplot as plt
from yaml import load

import mpl_setting
import util


def vclean(l_files, l_targets, xaxis=1, start=None, end=None):
    """
        vclean(l_files, l_targets, usemin)

            Parameters
            ----------
            l_files : list of str

            l_targets : list of str

            usemin : bool, optional

            xaxis: int

            start : None, int

            end: None, int

    """
    # read in files
    origin_header = []
    origin_data = []
    for _f in l_files:
        headers, data = util.read_data_file(_f)
        origin_header.append(headers)
        origin_data.append(data)

    prefix = l_files[0]
    prefix = prefix[0:prefix.find('_')]

    maxlength = 0
    for _d in origin_data:
        if len(_d) > maxlength:
            maxlength = len(_d)

    # reorganize data
    for _t in l_targets:
        # generate the summary file
        sum_data = np.zeros((maxlength, len(origin_data) + 2))
        sum_header = ['t', 'V_m']

        for index_file in range(len(origin_header)):
            if len(origin_data[index_file]) == maxlength:
                sum_data[:, 0] = origin_data[index_file]['t']
                sum_data[:, 1] = origin_data[index_file]['V_m']

            if _t not in origin_header[index_file]:
                print('Error: No data for target:{} in the data file. Exit.'.format(_t))
                exit(1)

            sum_data[:len(origin_data[index_file]), index_file + 2] = origin_data[index_file][_t]

            _h = l_files[index_file].replace('.dat', '')
            for i in range(xaxis-1):
                _h = _h[:_h.rfind('_')]
            _h = _h[_h.rfind('_')+1:]
            sum_header.append(_h)

        summary_file = _t + '_' + prefix + '_summary.dat'
        util.save_data_file(summary_file, sum_header, sum_data)

        # generate the I-V relationship
        iv_header = ['V', 'I']
        iv_data = np.zeros((len(l_files), 2))

        import re
        for _i in range(2, len(sum_header)):
            V = sum_header[_i]
            V = re.sub(r'[a-zA-Z]', '', V)
            V = float(V)

            if start is not None and end is not None:
                max_index = np.argmax(np.abs(sum_data[start:end, _i]))
            else:
                start = 0
                max_index = np.argmax(np.abs(sum_data[:, _i]))

            I = sum_data[start+max_index, _i]
            iv_data[_i-2] = V, I

        iv_data = iv_data[iv_data[:, 0].argsort()]  # sort by the first column

        iv_file = _t + '_' + prefix + '_IV.dat'
        util.save_data_file(iv_file, iv_header, iv_data)


def plotvc(vc_yaml, start=1, end=0, iftext=True):
    """
    Plot voltage traces of a voltage clamp protocol.

    :param vc_yaml:
    :param start: Points truncated from the start.
    :param end: Points truncated from the end.
    :return:
    """
    mpl_setting.set_matplotlib_default()
    config = load(vc_yaml)

    plt.figure(figsize=[mpl_setting.fig_size[0]/4, mpl_setting.fig_size[1]/8])

    # must be wrapped by 'Voltage_Clamp' and 'Protocol'
    config = config['Voltage_Clamp']['Protocol']

    holding = [config['HOLDING_V']] * config['HOLDING_TIME']
    prepulse = [config['PREPULSE_V']] * config['PREPULSE_TIME']
    postpulse = [config['POSTPULSE_V']] * config['POSTPULSE_TIME']
    tail = [config['TAIL_V']] * config['TAIL_TIME']

    time = range(config['HOLDING_TIME'] + config['PREPULSE_TIME'] + config['DURATION'] + config['POSTPULSE_TIME'] + config['TAIL_TIME'])

    test_v = np.arange(config['VC_START'], config['VC_END'], config['VC_STEP'])
    for v in test_v:
        test = [v] * config['DURATION']
        voltage = holding + prepulse + test + postpulse + tail
        plt.plot(time, voltage, 'k')

    ax = plt.gca()
    if iftext:
        # text on test voltage
        ax.text(x=config['HOLDING_TIME'] + config['PREPULSE_TIME'] - 10,
                y=test_v[0],
                s='{:.0f} mV'.format(test_v[0]),
                va='center', ha='right')
        ax.text(x=config['HOLDING_TIME'] + config['PREPULSE_TIME'] - 10,
                y=test_v[-1],
                s='{:.0f} mV'.format(test_v[-1]),
                va='center', ha='right')
        ax.text(x=config['HOLDING_TIME'] + config['PREPULSE_TIME'] + 0.5 * config['DURATION'],
                y=test_v[-1] + 1,
                s='{:.0f} ms'.format(config['DURATION']),
                va='bottom', ha='center')

        # text on prepulse and postpulse
        if config['PREPULSE_TIME'] >= 200:
            ax.text(x=config['HOLDING_TIME'] - 10,
                    y=config['PREPULSE_V'],
                    s='{:.0f} mV'.format(config['PREPULSE_V']),
                    va='center', ha='right')
            ax.text(x=config['HOLDING_TIME'] + 0.5 * config['PREPULSE_TIME'],
                    y=config['PREPULSE_V'] + 1,
                    s='{:.0f} ms'.format(config['PREPULSE_TIME']),
                    va='bottom', ha='center')
        if config['POSTPULSE_TIME'] >= 200:
            ax.text(x=config['HOLDING_TIME'] + config['PREPULSE_TIME'] + config['DURATION'] + config['POSTPULSE_TIME'] + 10,
                    y=config['POSTPULSE_V'],
                    s='{:.0f} mV'.format(config['POSTPULSE_V']),
                    va='center', ha='left')
            ax.text(x=config['HOLDING_TIME'] + config['PREPULSE_TIME'] + config['DURATION'] + 0.5 * config['POSTPULSE_TIME'],
                    y=config['POSTPULSE_V'] + 1,
                    s='{:.0f} ms'.format(config['POSTPULSE_TIME']),
                    va='bottom', ha='center')

        # text on holding and tailing
        ax.text(x=start,
                y=config['HOLDING_V'] + 1,
                s='{:.0f} mV'.format(config['HOLDING_V']),
                va='bottom', ha='left')
        if config['TAIL_TIME'] >= 200:
            ax.text(x=time[-1] - end,
                    y=config['TAIL_V'] + 1,
                    s='{:.0f} mV'.format(config['TAIL_V']),
                    va='bottom', ha='right')

    # reset xlimit
    ax.set_xlim(start, time[-1] - end)

    # remove axis
    ax.set_axis_off()

    plt.savefig('voltage.png')
    plt.show()


if __name__ == '__main__':
    plotvc(__doc__)
