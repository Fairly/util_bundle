"""
Voltage_Clamp:
 DT: 0.02

 Target: I_CaT
 Protocol:
   HOLDING_V: -90
   HOLDING_TIME: 200

   # pre-pulse means a period between holding and test
   PREPULSE_V: -60
   PREPULSE_TIME: 300

   # test potential
   VC_START: -40
   VC_END: 11.0
   VC_STEP: 10.0
   DURATION: 200 # test_time

   TAIL_V: -90
   TAIL_TIME: 200
"""
import numpy as np
import matplotlib.pyplot as plt
import yaml

import mpl_setting


def plotvc(vc_yaml, start=1, end=0):
    """

    :param vc_yaml:
    :param start: Points truncated from the start.
    :param end: Points truncated from the end.
    :return:
    """
    mpl_setting.set_matplotlib_default()
    config = yaml.load(vc_yaml)

    # must be wrapped by 'Voltage_Clamp' and 'Protocol'
    config = config['Voltage_Clamp']['Protocol']

    holding = [config['HOLDING_V']] * config['HOLDING_TIME']
    prepulse = [config['PREPULSE_V']] * config['PREPULSE_TIME']
    tail = [config['TAIL_V']] * config['TAIL_TIME']

    time = range(config['HOLDING_TIME'] + config['PREPULSE_TIME'] + config['DURATION'] + config['TAIL_TIME'])

    test_v = np.arange(config['VC_START'], config['VC_END'], config['VC_STEP'])
    for v in test_v:
        test = [v] * config['DURATION']
        voltage = holding + prepulse + test + tail
        plt.plot(time, voltage, 'k')

    ax = plt.gca()
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

    # text on holding and tailing
    ax.text(x=start,
            y=config['HOLDING_V'] + 1,
            s='{:.0f} mV'.format(config['HOLDING_V']),
            va='bottom', ha='left')
    ax.text(x=time[-1] - end,
            y=config['TAIL_V'] + 1,
            s='{:.0f} mV'.format(config['TAIL_V']),
            va='bottom', ha='right')

    # reset xlimit
    ax.set_xlim(start, time[-1] - end)

    # remove axis
    ax.set_axis_off()

    plt.show()


if __name__ == '__main__':
    plotvc(__doc__)
