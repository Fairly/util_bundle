#! /usr/bin/python
# coding=utf-8
"""

"""

import re
import sys
from pprint import pprint

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from numpy import ceil, floor

from util_bundle.util import read_data_file


def get_axes_num(n):
    sqr_n = np.sqrt(n)
    if sqr_n.is_integer():
        return int(sqr_n), int(sqr_n)
    else:
        r = int(ceil(sqr_n))
        c = int(floor(sqr_n))
        while r * c < n:
            r += 1
        return r, c


def iflastrow(column, i, num_of_axes=None):
    if i > num_of_axes - column:
        return True
    else:
        return False


def catagrize_data(data):
    """
    This function is used to divide the data by their contents. For example,
    for a result got from a running of cell model, there will be data fields
    describing currents, potential, concentrations and so on.
    This function separates them and records their label name in `data`,
    in order to match the flags in the `autoplot` function.

    :param data:
    :return:
    """
    keywords = ['I', 'Ca', 'Na', 'K', 'CaMKII', 'V', 'J', 'CaM', 'RyR', 'Phos']

    re_tmp = re.compile(r'([A-Z][a-zA-Z]*)(?:_([a-zA-Z]*))*')
    for j, label in enumerate(data['l_labels']):
        m = re.match(re_tmp, label)
        if m and m.group(1) in keywords:
            if m.group(1) not in data.keys():
                data[m.group(1)] = []
            data[m.group(1)].append(label)

    return data


def gen_data(l_input_filename, l_content_name):
    """
    This function is not stable now.
    The first file in parameters should be the major file to plot. All data in the major file will
    be plotted, those in other files may not.

    :param l_input_filename:
    :param l_content_name: list
                这个里面存的名字要和filename的文件内容对应，就是一个文件对应一条曲线的Label。
    """
    data = []
    for i, filename in enumerate(l_input_filename):
        l_labels, npa_data = read_data_file(filename)
        data.append({'data': npa_data,
                     'l_labels': l_labels,
                     'name': l_content_name[i]})

        for label in data[i]['l_labels']:
            if label.startswith(('t', 'T')):
                data[i]['xaxis'] = label
                break

        if len(data[i]['l_labels']) < len(data[i]['data'][0]):
            raise Exception("\nErr: The data file named %s has something wrong.\n" % filename +
                            "Err: The length of its header is less than the length of data fields.")

    data[0] = catagrize_data(data[0])

    return data


def my_plot(data, l_labels, xlimit=None):
    fig = plt.figure()

    num_of_axes = len(l_labels)
    r, c = get_axes_num(num_of_axes)

    legend_flag = False
    for i, column_label in enumerate(l_labels):
        if column_label == data[0]['xaxis']:
            continue

        axe = fig.add_subplot(r, c, i + 1)

        for d in data:
            if column_label in d['l_labels']:
                axe.plot(d['data'][d['xaxis']],
                         d['data'][column_label],
                         label=d['name'])

        if not iflastrow(c, i + 1, num_of_axes):
            axe.get_xaxis().set_visible(False)
        axe.set_title(column_label)
        if xlimit is not None:
            axe.set_xlim(*xlimit)
        if not legend_flag:
            axe.legend(prop={'size': 10})
            legend_flag = True


def autoplot(l_input_filename, l_label, flags=('all',), xlimit=None, outfigname=None):
    """
    

    :param l_input_filename:
    :param l_label:
    :param flags:
    :param xlimit: None, tuple or a tuple of two list
    :param outfigname: If no fig name is given, no fig will be saved.
    """
    data = gen_data(l_input_filename, l_label)

    # translation data to align them
    if xlimit is not None and isinstance(xlimit[0], list):
        start, end = xlimit
        new_end = []
        for s, e, d in zip(start, end, data):
            d['data'][d['xaxis']] = d['data'][d['xaxis']] - s  # translation the 'xaxis' column of d['data']
            new_end.append(e-s)
        max_end = max(new_end)
        xlimit = (0, max_end)

    if 'all' in flags:
        if len(data[0]['l_labels']) < 10:
            my_plot(data, data[0]['l_labels'].remove(data[0]['xaxis']), xlimit)
        else:
            my_plot(data, data[0]['V'] + data[0]['I'], xlimit)
            my_plot(data, set(data[0]['l_labels']) - set(data[0]['V']) - set(data[0]['I']), xlimit)
    else:
        l_gca = []
        for flag in flags:
            if len(l_gca) < 8:
                if flag in data[0]['l_labels']:
                    l_gca.append(flag)
                else:
                    l_gca += data[0][flag]
                continue
            my_plot(data, l_gca, xlimit)
            if flag in data[0]['l_labels']:
                l_gca = [flag]
            else:
                l_gca = data[0][flag]
        my_plot(data, l_gca, xlimit)

    if outfigname is not None:
        plt.savefig(outfigname)
    else:
        plt.show()


def ros_cell_plot(l_input_filenames, l_legends):
    d = gen_data(l_input_filenames, l_legends)

    fig = plt.figure(1, figsize=(3.24, 4), dpi=300)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # fig.text(0.02, 0.93, 'A')

    gs = gridspec.GridSpec(5, 2)
    gs.update(left=0.15, right=0.95, wspace=0.25)

    plt.subplot(gs[0, 0])
    plt.plot(d[2]['data'][:, 0],
             d[2]['data'][:, 1],
             '-', linewidth=1, label=r'$\mathrm{CaMKII}$')
    plt.plot(d[3]['data'][:, 0],
             d[3]['data'][:, 1],
             '--', linewidth=1, label=r'$\mathrm{Control}$')
    plt.ylabel(r'$\mathrm{Vm}\,(mV)$', fontsize=6)
    plt.ylim((-80, 40))
    plt.xlim((58900, 59600))
    labels = ['', '0', '', '200', '', '400', '', '600']
    plt.xticks([58900, 59000, 59100, 59200, 59300, 59400, 59500, 59600], labels)
    plt.yticks(np.arange(-80, 41, 40))
    plt.gca().set_xticklabels([])
    plt.legend(fontsize=6, frameon=False)

    plt.subplot(gs[0, 1])
    plt.plot(d[0]['data'][:, 0],
             d[0]['data'][:, 1],
             '-', linewidth=1, label='CaMKII')
    plt.plot(d[1]['data'][:, 0],
             d[1]['data'][:, 1],
             '--', linewidth=1, label='Control')
    plt.ylim((-80, 40))
    plt.xlim((59650, 60000))
    labels = ['', '0', '', '100', '', '200', '', '300']
    plt.xticks([59650, 59700, 59750, 59800, 59850, 59900, 59950, 60000], labels)
    plt.yticks(np.arange(-80, 41, 40))
    plt.gca().set_xticklabels([])

    plt.subplot(gs[1, 0])
    plt.plot(d[2]['data'][:, 0],
             d[2]['data'][:, 2],
             '-', linewidth=1, label='CaMKII')
    plt.plot(d[3]['data'][:, 0],
             d[3]['data'][:, 2],
             '--', linewidth=1, label='Control')
    plt.ylabel(r'$\mathrm{I_{Na}}\,(A/F)$', fontsize=6, labelpad=1.5)
    plt.ylim((-100, 10))
    plt.xlim((58900, 59600))
    plt.yticks(np.arange(-100, 21, 40))
    labels = ['', '0', '', '200', '', '400', '', '600']
    plt.xticks([58900, 59000, 59100, 59200, 59300, 59400, 59500, 59600], labels)
    plt.gca().set_xticklabels([])

    plt.subplot(gs[1, 1])
    plt.plot(d[0]['data'][:, 0],
             d[0]['data'][:, 2],
             '-', linewidth=1, label='CaMKII')
    plt.plot(d[1]['data'][:, 0],
             d[1]['data'][:, 2],
             '--', linewidth=1, label='Control')
    plt.ylim((-100, 10))
    plt.xlim((59650, 60000))
    plt.yticks(np.arange(-100, 21, 40))
    labels = ['', '0', '', '100', '', '200', '', '300']
    plt.xticks([59650, 59700, 59750, 59800, 59850, 59900, 59950, 60000], labels)
    plt.gca().set_xticklabels([])

    plt.subplot(gs[2, 0])
    plt.plot(d[2]['data'][:, 0],
             d[2]['data'][:, 3],
             '-', linewidth=1, label='CaMKII')
    plt.plot(d[3]['data'][:, 0],
             d[3]['data'][:, 3],
             '--', linewidth=1, label='Control')
    plt.ylabel(r'$\mathrm{I_{CaL}}\,(A/F)$', fontsize=6)
    plt.xlim((58900, 59600))
    plt.yticks(np.arange(-10, 2.1, 4))
    labels = ['', '0', '', '200', '', '400', '', '600']
    plt.xticks([58900, 59000, 59100, 59200, 59300, 59400, 59500, 59600], labels)
    plt.gca().set_xticklabels([])

    plt.subplot(gs[2, 1])
    plt.plot(d[0]['data'][:, 0],
             d[0]['data'][:, 3],
             '-', linewidth=1, label='CaMKII')
    plt.plot(d[1]['data'][:, 0],
             d[1]['data'][:, 3],
             '--', linewidth=1, label='Control')
    # plt.ylim((-100, 10))
    plt.xlim((59650, 60000))
    plt.yticks(np.arange(-10, 2.1, 4))
    labels = ['', '0', '', '100', '', '200', '', '300']
    plt.xticks([59650, 59700, 59750, 59800, 59850, 59900, 59950, 60000], labels)
    plt.gca().set_xticklabels([])

    plt.subplot(gs[3, 0])
    plt.plot(d[2]['data'][:, 0],
             d[2]['data'][:, 11],
             '-', linewidth=1, label='CaMKII')
    plt.plot(d[3]['data'][:, 0],
             d[3]['data'][:, 11],
             '--', linewidth=1, label='Control')
    plt.ylabel(r'$\mathrm{J_{Rel}}\,(A/F)$', fontsize=6)
    plt.ylim((0, 0.04))
    plt.xlim((58900, 59600))
    plt.yticks(np.arange(0, 0.041, 0.020))
    labels = ['', '0', '', '200', '', '400', '', '600']
    plt.xticks([58900, 59000, 59100, 59200, 59300, 59400, 59500, 59600], labels)
    plt.gca().set_xticklabels([])

    plt.subplot(gs[3, 1])
    plt.plot(d[0]['data'][:, 0],
             d[0]['data'][:, 11],
             '-', linewidth=1, label='CaMKII')
    plt.plot(d[1]['data'][:, 0],
             d[1]['data'][:, 11],
             '--', linewidth=1, label='Control')
    # plt.ylim((-100, 10))
    plt.xlim((59650, 60000))
    plt.yticks(np.arange(0, 0.041, 0.020))
    labels = ['', '0', '', '100', '', '200', '', '300']
    plt.xticks([59650, 59700, 59750, 59800, 59850, 59900, 59950, 60000], labels)
    plt.gca().set_xticklabels([])

    plt.subplot(gs[4, 0])
    plt.plot(d[2]['data'][:, 0],
             d[2]['data'][:, 22],
             '-', linewidth=1, label='CaMKII')
    plt.plot(d[3]['data'][:, 0],
             d[3]['data'][:, 15],
             '--', linewidth=1, label='Control')
    plt.ylabel(r'$\mathrm{[Ca^{2+}]_{junc}}\,(mM)$', fontsize=6)
    plt.xlabel(r'$\mathrm{t}\;(ms)$', fontsize=8)
    plt.ylim((0, 0.06))
    plt.xlim((58900, 59600))
    plt.yticks(np.arange(0, 0.061, 0.030))
    labels = ['', '0', '', '200', '', '400', '', '600']
    plt.xticks([58900, 59000, 59100, 59200, 59300, 59400, 59500, 59600], labels)

    plt.subplot(gs[4, 1])
    plt.plot(d[0]['data'][:, 0],
             d[0]['data'][:, 22],
             '-', linewidth=1, label='CaMKII')
    plt.plot(d[1]['data'][:, 0],
             d[1]['data'][:, 15],
             '--', linewidth=1, label='Control')
    # plt.ylim((-100, 10))
    plt.xlim((59650, 60000))
    plt.xlabel(r'$\mathrm{t}\;(ms)$', fontsize=8)
    plt.yticks(np.arange(0, 0.061, 0.030))
    labels = ['', '0', '', '100', '', '200', '', '300']
    plt.xticks(np.arange(59650, 60050, 50), labels)
    # plt.xlabel(r'$\mathrm{[CaMKII]_{total}}\;(mM)$', fontsize=8, labelpad=2)
    # plt.ylabel(r'$\mathrm{CaM\;bound\;fraction}\;(\%)$', fontsize=8)
    # plt.axes([0.15, 0.18, 0.8, 0.77])
    # plt.semilogx(x, fit_hillequation(x, *curve_5a) * 100, '--', linewidth=1)

    # fig.add_subplot(2, 2, 2)
    # plt.plot(d[0]['data'][:, 0],
    #          d[0]['data'][:, 14],
    #          '-', linewidth=1, label='CaMKII')
    # plt.tick_params(axis='both', which='major', labelsize=8)
    # plt.ylim((0.2, 0.3))
    # plt.xlim((59600, 60000))

    # fig.add_subplot(2, 2, 4)
    # plt.plot(d[2]['data'][:, 0],
    #          d[2]['data'][:, 14],
    #          '-', linewidth=1, label='CaMKII')
    # plt.tick_params(axis='both', which='major', labelsize=8)
    # plt.ylim((0, 0.1))
    # plt.xlim((58900, 59700))

    for ax in fig.axes:
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='both', which='major', labelsize=6)

    plt.savefig('/Users/fairly/Desktop/8-23-ap.pdf')
    plt.show()


def tmpplot():
    with open('../Grandi_C_2011/outputall_500.dat', 'r') as f_tmp:
        npa_data = np.genfromtxt(f_tmp)

    fig = plt.figure()
    plt.subplot(221)
    plt.plot(npa_data[:, 0], npa_data[:, 39])

    plt.subplot(222)
    plt.plot(npa_data[:, 0], npa_data[:, 17])

    plt.subplot(223)
    plt.plot(npa_data[:, 0], npa_data[:, 18])

    plt.subplot(224)
    plt.plot(npa_data[:, 0], npa_data[:, 19])

    for ax in fig.axes:
        ax.set_xlim([58800, 60000])

    plt.show()


def tmpplot1():
    with open('../OHara/output.txt', 'r') as f_tmp:
        f_tmp.readline()
        npa_data = np.genfromtxt(f_tmp)

    plt.figure()
    plt.subplot(221)
    plt.plot(npa_data[:, 0], npa_data[:, 1])

    plt.subplot(222)
    plt.plot(npa_data[:, 0], npa_data[:, 19])

    #     plt.subplot(223)
    #     plt.plot(npa_data[:, 0], npa_data[:, 18])

    #     plt.subplot(224)
    #     plt.plot(npa_data[:, 0], npa_data[:, 19])

    #     for ax in fig.axes:
    #         ax.set_xlim([58800, 60000])

    plt.show()


def iv_plot():
    heads, data = read_data_file("../bin/vc/summary.dat")
    plt.plot(data[:, 0], data[:, 1] / max(abs(data[:, 1])))
    plt.xlabel("Voltage (mV)", fontsize=12)
    plt.ylabel("Normalised INa", fontsize=12)
    plt.savefig('/Users/fairly/Desktop/abc/summary_INa.pdf')
    plt.show()


def current_plot():
    heads, data = read_data_file("../cmake-build-debug/bin/output/20161231-233450/result.dat")
    for i in range(1, len(heads)):
        plt.plot(data[190:240, 0], data[190:240, i])

    # plt.savefig('/Users/fairly/Desktop/abc/current_INa.pdf')
    plt.show()


if __name__ == '__main__':
    #     iv_plot()
    #    current_plot()
    autoplot(  # [ "/Users/fairly/Documents/workspace_cpp/weijian_origin/build/debug/bin/output/out_atria_WT"
        # "/37C_CTL_BCL-500.dat"],
        ["/Users/fairly/NutBox/实验室/报告/2017/4-AP Block/37C_CTL_BCL-250.dat",
         "/Users/fairly/NutBox/实验室/报告/2017/4-AP Block/4AP_Block_%30Ito_%100IKur_bcl250.dat"],
        ["CTL", "+ 4-AP"],
        flags=('V',),
        xlimit=(0, 60),
        # outfigname="/Users/fairly/Desktop/abc/V_and_I_atria_ventricle.pdf"
    )
