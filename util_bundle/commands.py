import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from subprocess import call

from docopt import docopt
from schema import Schema, Or, Use, And
import numpy as np
import time
# from scipy.misc import imread, imsave, imresize
# from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
import skimage
import skimage.filters
import skimage.draw
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from util import read_data_file, save_data_file, bar_arrange, write_scalar_vtk
from util_bundle.measurement import dt_recognition, ap_recognition, get_range
import measurement
import plot as myplot

CELL_MODEL = ''  # used in some commands


class AbstractCommand:
    """Base class for the commands"""

    def __init__(self, command_args, global_args):
        """
        Initialize the commands.
        :param command_args: arguments of the command
        :param global_args: arguments of the program
        """
        self.args = docopt(str(self.__doc__), argv=command_args)
        self.global_args = global_args

    def execute(self):
        """Execute the commands"""
        raise NotImplementedError


class measure(AbstractCommand):
    """
usage:  measure    [-m] (-d=DIR | <FILE>...)

Options:
    -m          Open multi-processing.
    -d=DIR      Process all files written by cell models under DIR.
Arguments:
    <FILE>...   Files to be processed.
    """

    def execute(self):
        schema = Schema({
            '-d': Or(None, os.path.isdir),
            '-m': bool,
            '<FILE>': Or(None, [os.path.isfile], error='Cannot find file[s].'),
        }
        )

        args = schema.validate(self.args)

        p = ThreadPoolExecutor(cpu_count()) if args['-m'] else ThreadPoolExecutor(1)
        if args['-d']:
            # given a directory, measure all files in it
            onlyfiles = [os.path.join(args['-d'], f)
                         for f in os.listdir(args['-d'])
                         if os.path.isfile(os.path.join(args['-d'], f))]
            current_files = [f for f in onlyfiles if 'currents' in f]

            p.map(measurement.measure, current_files)
        else:
            # given a list of files
            for f in args['<FILE>']:
                measurement.measure(f)


class clean(AbstractCommand):
    """
usage: clean [options] <FILE>...
       clean -R <BACKUPFILE>...

Clean the data file. Original file will be backed up.

Options:
    -t=num, --truncate=num
                Whether truncate the result file to tailing `num` of rows.
    -s=dt, --shrink=dt
                Use a new dt bigger than the one in data files to shrink it.
    -u --underline
                Whether add a '_' mark to every field in header names.
    -r=time --reset-time=time
                If the start time is not 'time', reset it to 'time'.
    -R --recover
                If regret, try this. Recover from backup files.
Arguments:
    <FILE>...          Files to be processed.
    <BACKUPFILE>...    Names of backup files.
    """

    def execute(self):
        schema = Schema({'--reset-time': Or(None, Use(float)),
                         '--truncate': Or(None, Use(int)),
                         '--shrink': Or(None, Use(float)),
                         '--underline': bool,
                         '--recover': bool,
                         '<FILE>': Or(None, [os.path.isfile], error='Cannot find file[s].'),
                         '<BACKUPFILE>': Or(None, [os.path.isfile], error='Cannot find file[s].'),
                         }
                        )

        args = schema.validate(self.args)

        def clean_result_for_plot(filename, add_underline=False, truncate_to=None, shrink=None,
                                  reset_start_time=False, tail=True):  # TODO parameter `tail`
            """
            When plotting a result, it's common to reduce the size of the result file first.
            """
            from subprocess import call
            backup_file_name = filename + '.backup.dat'
            tmp_file_name = 'tmp.dat'

            shutil.copyfile(filename, backup_file_name)  # backup
            shutil.move(filename, tmp_file_name)

            if truncate_to is not None:
                # use shell command is usually faster than python itself
                to_file = open(filename, mode="w")
                call(['head', '-n', '1', tmp_file_name], stdout=to_file)
                to_file.close()

                to_file = open(filename, mode="a")
                call(['tail', '-n', str(truncate_to), tmp_file_name], stdout=to_file)
                to_file.close()

                shutil.move(filename, tmp_file_name)

            if shrink is not None:
                _, data = read_data_file(tmp_file_name, max_rows=2)  # read only 2 data lines to speed up
                dt = dt_recognition(data)
                multiplier = int(shrink / dt)

                to_file = open(filename, mode="w")
                # save the first and second line, then every `multiplier`th line to file
                call(['awk', 'NR == 1 || NR ==2 || (NR-2) % ' + str(multiplier) + ' == 0', tmp_file_name],
                     stdout=to_file)
                to_file.close()

                shutil.move(filename, tmp_file_name)

            if add_underline:
                headers, _ = read_data_file(tmp_file_name, max_rows=2)  # read only 2 data lines to speed up

                if not headers[0].endswith('_'):
                    for i, tag in enumerate(headers):
                        headers[i] = tag + '_'

                    # these code seems cumbersome, but fast!
                    from_file = open(tmp_file_name)
                    from_file.readline()
                    to_file = open(filename, mode="w")
                    to_file.write('\t'.join(headers))
                    shutil.copyfileobj(from_file, to_file)

                    shutil.move(filename, tmp_file_name)

            if reset_start_time:
                headers, data = read_data_file(tmp_file_name)
                dt = dt_recognition(data)
                if data[0][0] == reset_start_time:  # don't need to reset
                    pass
                else:
                    time_offset = reset_start_time - data[0][0]
                    data['t'] += time_offset
                    save_data_file(filename, header=headers, data=data)

                shutil.move(filename, tmp_file_name)

            # all operations done
            shutil.move(tmp_file_name, filename)

        if args['--recover']:
            for f in args['<BACKUPFILE>']:
                original_file = f.replace('.backup.dat', '')
                shutil.move(f, original_file)
        else:
            for f in args['<FILE>']:
                clean_result_for_plot(f, add_underline=args['--underline'],
                                      truncate_to=args['--truncate'],
                                      shrink=args['--shrink'],
                                      reset_start_time=args['--reset-time'])


class qplot(AbstractCommand):
    """
An quick interface for plotting and comparing APs or currents.

usage: qplot [options] [(-x <XSTART> <XEND>)] (<FILE> <LABEL>)...
       qplot [options] -s (<FSTART> <FEND> <FILE> <LABEL>)...

Options:
    -V          Trigger for whether plotting the AP.
    -I          Trigger for whether plotting currents.
    -A          Trigger for whether plotting all fields.
    -C=CTM      Customized plotting prefixes, separated by ','. For example,
                "V,I" means plotting fields whose name starts with "V" or "I".

    -o=OUT      The file name of the output figure.
                If not given, show the figure instead of saving it.

    -x          Whether set limits on the x axis.
    -s          Separately set x-limits of all FILEs.

Arguments:
    XSTART XEND
                The starting and ending points of x-limits.
    FILE LABEL
                One Label for one file.
    FSTART FEND
                The starting and ending points of x-limits of each FILE.
    """

    def execute(self):
        schema = Schema({
            '-I': bool,
            '-V': bool,
            '-A': bool,
            '-C': Or(None, And(str, len)),
            '-o': Or(None, And(str, len)),
            '-x': bool,
            '-s': bool,
            '<FILE>': [os.path.isfile],
            '<LABEL>': [str],
            '<XEND>': Or(None, Use(float)),
            '<XSTART>': Or(None, Use(float)),
            '<FEND>': Or(None, [Use(float)]),
            '<FSTART>': Or(None, [Use(float)]),
        })

        args = schema.validate(self.args)

        plot_flag = []
        if args['-A']:
            plot_flag.append('all')
        else:
            if args['-V']:
                plot_flag.append('V')
            if args['-I']:
                plot_flag.append('I')
            if args['-C']:
                plot_flag.extend(args['-C'].split(','))

        if not plot_flag:  # default: plot voltage
            plot_flag.append('V')

        plot_flag = list(set(plot_flag))  # remove duplicated items

        if args['-x']:
            xlim = (args['<XSTART>'], args['<XEND>'])
        elif args['-s']:
            xlim = (args['<FSTART>'], args['<FEND>'])
        else:
            xlim = None

        myplot.autoplot(args['<FILE>'], args['<LABEL>'],
                        flags=plot_flag, outfigname=args['-o'], xlimit=xlim)


class plotvc(AbstractCommand):
    """
Usage:  plotvc [-s=start] [-e=end] <YAMLFILE>

Options:
  -s=start    Set the start time of plotting, since the holding time is usually too long. [default: 1]
  -e=end      Set the number in mili-seconds truncated from the end. [default: 0]

Arguments:
  <YAMLFILE>  The config file containing the voltage clamp protocol.
    """

    def execute(self):
        schema = Schema({
            "-s": Use(int),
            "-e": Use(int),
            "<YAMLFILE>": os.path.isfile,
        }
        )

        args = schema.validate(self.args)

        import plotvc as pvc
        f = open(args["<YAMLFILE>"], 'r')
        pvc.plotvc(f, start=args['-s'], end=args['-e'])


class freq(AbstractCommand):
    """
Usage:  freq    [-m] [-s=STEP] [-c=CELLTYPE] [--] <SIMTIME>
                (<FREQUENCY> | (<FSTART> <FSTOP>) [<FSTEP>])

Options:
  -s=STEP --step=STEP
                The time STEP (in unit ms) used to run the cell model [default: 0.1].
  -c=CELLTYPE --celltype=CELLTYPE
                Specify a cell type. 1 for atria, 0 for ventricles. [default: 1]

Arguments:
  SIMTIME       Specify how long the simulation is (unit second).
  FREQUENCY     An integer for running only one frequency.
  FSTART FSTOP FSTEP
                Specify a range of frequencies.
    """

    def execute(self):
        schema = Schema({
            '--step': Use(float),
            '--celltype': Use(int),
            '<SIMTIME>': Or(None, Use(float)),
            '<FREQUENCY>': Or(None, Use(float), error='FREQUENCY must be a number.'),
            '<FSTART>': Or(None, Use(int)),
            '<FSTOP>': Or(None, Use(int)),
            '<FSTEP>': Or(None, Use(int)),
            '--': bool
        }
        )

        args = schema.validate(self.args)

        if args['<FREQUENCY>']:
            # if only one frequency is specified in args, change it into a range
            # format
            f = [args['<FREQUENCY>'], args['<FREQUENCY>'] + 1]
        elif args['<FSTEP>']:
            f = [args['<FSTART>'], args['<FSTOP>'], args['<FSTEP>']]
        else:
            f = [args['<FSTART>'], args['<FSTOP>']]

        params = []
        for _freq in np.arange(*f):
            params.append(["./" + CELL_MODEL,
                           str(int(1000 / _freq)),
                           str(int(args['<SIMTIME>'] * _freq)),
                           str(args['--step'])])
        # Multi-process
        p = ThreadPoolExecutor(cpu_count()) if args['-m'] else ThreadPoolExecutor(1)
        p.map(call, params)


class burst(AbstractCommand):
    """
Usage:  burst    [-s=STEP] [-c=CELLTYPE] [--]
                 <BCL> <BEATS> <BURST_BCL> <BURST_BEATS> [<REST_BCL> <REST_BEATS>]

Options:
  -s=STEP --step=STEP
                The time STEP (in unit ms) used to run the cell model [default: 0.1].
  -c=CELLTYPE --celltype=CELLTYPE
                Specify a cell type. 1 for atria, 0 for ventricles. [default: 1]

Arguments:
  BCL           An integer of BCL(unit msec). [default: 1000]
  BEATS         How many beats a simulation has.
  BURST_BCL BURST_BEATS REST_BCL REST_BEATS
                Specify the burst protocol.
    """

    def execute(self):
        schema = Schema({
            '--step': Use(float),
            '--celltype': Use(int),
            '<BCL>': Or(None, Use(float)),
            '<BEATS>': Or(None, Use(int)),
            '<BURST_BCL>': Or(None, Use(float)),
            '<BURST_BEATS>': Or(None, Use(int)),
            '<REST_BCL>': Or(None, Use(float)),
            '<REST_BEATS>': Or(None, Use(int)),
            '--': bool
        }
        )

        args = schema.validate(self.args)

        global CELL_MODEL
        call(['./' + CELL_MODEL,
              str(args['<BCL>']),
              str(args['<BEATS>']),
              str(args['--step']),
              str(args['<BURST_BCL>']),
              str(args['<BURST_BEATS>']),
              str(args['<REST_BCL>']),
              str(args['<REST_BEATS>'])])


class s1s2(AbstractCommand):
    """
Usage:  s1s2    [-s=STEP] [-c=CELLTYPE] <BCL> <BEATS> <S1S2SPAN>

Options:
  -s=STEP --step=STEP
                The time STEP (in unit ms) used to run the cell model [default: 0.1].
  -c=CELLTYPE --celltype=CELLTYPE
                Specify a cell type. 1 for atria, 0 for ventricles. [default: 1]

Arguments:
  BCL           An integer of BCL(unit msec). [default: 1000]
  BEATS         How many beats a simulation has.
  S1S2SPAN      The time span between normal S1 stimuli and the S2 stimulus.
    """

    def execute(self):
        schema = Schema({
            '--step': Use(float),
            '--celltype': Use(int),
            '<BCL>': Or(None, Use(float)),
            '<BEATS>': Or(None, Use(int)),
            '<S1S2SPAN>': Or(None, Use(float)),
        }
        )

        args = schema.validate(self.args)

        global CELL_MODEL
        call(['./' + CELL_MODEL,
              str(args['<BCL>']),
              str(args['<BEATS>']),
              str(args['--step']),
              str(args['<S1S2SPAN>'])])


class run(AbstractCommand):
    """
usage:  run  [-m] [-s=STEP] [-c=CELLTYPE] [--] <SIMTIME>
             (<BCL> | <BSTART> <BSTOP> [<BSTEP>])

Options:
  -m            Open multi-processing.
  -s=STEP --step=STEP
                The time STEP (in unit ms) used to run the cell model [default: 0.1].
  -c=CELLTYPE --celltype=CELLTYPE
                Specify a cell type. 1 for atria, 0 for ventricles. [default: 1]

Arguments:
  SIMTIME       Specify how long the simulation is (unit second).
  BCL           An integer of BCL(unit msec). [default: 1000]
  BSTART BSTOP BSTEP
                Specify a range of BCLs.
    """

    # TODO update with smt?
    def execute(self):
        schema = Schema({
            '--step': Use(float),
            '--celltype': Use(int),
            '<SIMTIME>': Or(None, Use(float)),
            '<BCL>': Or(None, Use(float)),
            '<BSTART>': Or(None, Use(int)),
            '<BSTOP>': Or(None, Use(int)),
            '<BSTEP>': Or(None, Use(int)),
            '--': bool,
            '-m': bool,
        }
        )

        args = schema.validate(self.args)

        global CELL_MODEL
        # cell models are compiled from c using conditional compilation,
        # set the global model variable based on args.
        # TODO update
        if args['--celltype'] == 0:
            CELL_MODEL = 'ventri_cell_model'
        else:
            CELL_MODEL = 'atria_cell_model'
        if args['s1s2']:
            CELL_MODEL += '_s1s2'
        if args['burst']:
            CELL_MODEL += '_burst'

        if args['<BCL>']:
            # change BCL into a range format, for easy-to-use in the for loop below
            f = [args['<BCL>'], args['<BCL>'] + 1]
        elif args['<BSTEP>']:
            f = [args['<BSTART>'], args['<BSTOP>'], args['<BSTEP>']]
        else:
            f = [args['<BSTART>'], args['<BSTOP>']]

        params = []
        for bcl in np.arange(*f):
            params.append(["./" + CELL_MODEL,
                           str(bcl),
                           str(int(args['<SIMTIME>'] * 1000 / bcl)),
                           str(args['--step'])])
        # Multi-process
        p = ThreadPoolExecutor(cpu_count()) if args['-m'] else ThreadPoolExecutor(1)
        p.map(call, params)


class plot(AbstractCommand):
    """
Usage:
  plot  (--figtable | ([--save] (<ChapterNo> <FigureNo> | <FigNoinPaper>)))

Options:
  --figtable    Show numbers and names of figures.
  --save        Whether or not save the figure as a file.

Arguments:
  ChapterNo FigureNo
                The chapter and figure No. in Weijian's thesis.
  FigNoinPaper  The No. used in final mouse atrial cell model paper.
    """

    def __init__(self, command_args, global_args):
        AbstractCommand.__init__(self, command_args, global_args)

        # Matplotlib Style Sheet #
        # print mpl.rcParams.keys()
        fontsize = 10
        mpl.rcParams['font.family'] = 'Arial'
        mpl.rcParams['font.size'] = fontsize
        mpl.rcParams['font.weight'] = 'bold'

        # mpl.rcParams['text.usetex'] = True

        mpl.rcParams['mathtext.default'] = 'regular'

        mpl.rcParams['figure.figsize'] = 5.5, 4
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['figure.subplot.bottom'] = 0.18

        mpl.rcParams['axes.labelsize'] = fontsize
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['axes.labelweight'] = 'bold'

        mpl.rcParams['lines.linewidth'] = 1.5
        mpl.rcParams['lines.dash_capstyle'] = 'round'
        mpl.rcParams['lines.solid_capstyle'] = 'round'

        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['xtick.major.pad'] = 6
        mpl.rcParams['xtick.major.width'] = 1.5
        mpl.rcParams['ytick.labelsize'] = fontsize
        mpl.rcParams['ytick.major.pad'] = 6
        mpl.rcParams['ytick.major.width'] = 1.5

        mpl.rcParams['legend.frameon'] = False
        mpl.rcParams['legend.fontsize'] = fontsize - 1
        mpl.rcParams['legend.loc'] = 'best'
        mpl.rcParams['legend.handlelength'] = 3

        self.point = 1 / 72  # inch

        # Preferences for Plos Computational Biology
        self._fig_size = (7.5, 8.75)

        # Preferences for bar chars

    # ===============================================================================
    # celltype: (0, ventricle), (1, atria)
    # ===============================================================================
    @staticmethod
    def get_files(celltype=1, current=False, _measure=False):
        # Be aware of these paths. Change them if plotting OE or KO conditions.
        at_dir = 'output/out_atria_WT'
        ven_dir = 'output/out_ventricle_WT'

        rtn = []
        if celltype == 0:
            if current:
                rtn = [os.path.join(ven_dir, f) for f in os.listdir(ven_dir)
                       if os.path.isfile(os.path.join(ven_dir, f))
                       and 'currents' in f and 'burst' not in f and 's1s2' not in f]
            elif _measure:
                rtn = [os.path.join(ven_dir, f) for f in os.listdir(ven_dir)
                       if os.path.isfile(os.path.join(ven_dir, f))
                       and 'measure' in f and 'burst' not in f and 's1s2' not in f]
        elif celltype == 1:
            if current:
                rtn = [os.path.join(at_dir, f) for f in os.listdir(at_dir)
                       if os.path.isfile(os.path.join(at_dir, f))
                       and 'currents' in f and 'burst' not in f and 's1s2' not in f]
            elif _measure:
                rtn = [os.path.join(at_dir, f) for f in os.listdir(at_dir)
                       if os.path.isfile(os.path.join(at_dir, f))
                       and 'measure' in f and 'burst' not in f and 's1s2' not in f]

        # sort the file list with bcls in their names
        rtn.sort(cmp=lambda a, b: int(os.path.basename(a).split('_')[0]) - int(os.path.basename(b).split('_')[0]),
                 reverse=True)
        return rtn

    @staticmethod
    def to_percent(tick_value, position):
        # Ignore the passed-in position. This has the effect of scaling the default
        # tick locations.
        s = "{:.0f}".format(100 * tick_value)

        # The percent symbol needs escaping in latex
        if mpl.rcParams['text.usetex'] is True:
            return s + r'$\%$'
        else:
            return s + '%'

    def execute(self):
        schema = Schema({
            '--figtable': bool,
            '--save': bool,
            '<ChapterNo>': Or(None, Use(int)),
            '<FigureNo>': Or(None, Use(int)),
            '<FigNoinPaper>': Or(None, Use(int)),
        }
        )

        args = schema.validate(self.args)

        # If has '--figtable' flag, just show the figure table then return
        if args['--figtable']:
            with open('figure_table.txt', 'r') as fin:
                print(fin.read())
            return

        c_num = args['<ChapterNo>']
        f_num = args['<FigureNo>']
        f_num_inpaper = args['<FigNoinPaper>']

        figures = []

        # Figure 2.20, frequency #
        if c_num == 2 and f_num == 20:
            data = []
            for f in self.get_files(1, _measure=True):
                bcl = os.path.basename(f).split('_')[0]
                _freq = round(1000.0 / int(bcl))

                headers, tmpdata = myplot.read_data_file(f)

                apd90 = (tmpdata[-1][headers.index('APD_90')]
                         + tmpdata[-2][headers.index('APD_90')]) / 2
                apd75 = (tmpdata[-1][headers.index('APD_75')]
                         + tmpdata[-2][headers.index('APD_75')]) / 2
                apd50 = (tmpdata[-1][headers.index('APD_50')]
                         + tmpdata[-2][headers.index('APD_50')]) / 2
                apd25 = (tmpdata[-1][headers.index('APD_25')]
                         + tmpdata[-2][headers.index('APD_25')]) / 2

                data.append([_freq, apd90, apd75, apd50, apd25])

            data = np.array(data)
            # max_apd90 = max(data[:, 1])

            # for i in range(len(data)):
            # normalised
            # data[i][1] = data[i][1] / max_apd90

            # plot Figure a
            figures.append(plt.figure(1))
            axe = plt.subplot()
            axe.plot(data[:, 0], data[:, 1], 'k', linewidth=3)
            axe.set_xlim([0, 13])
            axe.set_ylabel('Normalised $APD_{90}$')
            axe.set_xlabel('Pacing Frequency (Hz)')

            # plot figure b
            figures.append(plt.figure(2))
            axe = plt.subplot()
            l, = axe.plot(data[:, 0], data[:, 2], '--k', label='$APD_{75}$')
            l.set_dashes([20, 5, 3, 5])
            l, = axe.plot(data[:, 0], data[:, 3], ':k', label='$APD_{50}$')
            l.set_dashes([10, 5])
            axe.plot(data[:, 0], data[:, 4], '-k', label='$APD_{25}$')
            axe.set_ylim([0, 22])
            axe.set_xlim([0, 13])
            axe.set_ylabel('Durations (ms)')
            axe.set_xlabel('Pacing Frequency (Hz)')
            axe.legend()

        # Figure 2.21 #
        elif c_num == 2 and f_num == 21:
            fig, axes = plt.subplots(nrows=2, ncols=2)

            # two file lists must have the same length
            f_ven_measures = self.get_files(0, _measure=True)
            f_at_measures = self.get_files(1, _measure=True)

            panel_1 = []
            panel_2 = []
            panel_3 = []
            panel_4 = []

            headers_v, tmpdata_v = myplot.read_data_file(f_ven_measures[0])
            i_ca_i_dia = headers_v.index('Ca_i_Dia')
            i_ca_i_max = headers_v.index('Ca_i_Max')
            i_ca_i_tau = headers_v.index('Ca_i_tau')
            i_na_dia = headers_v.index('Na_i_Dia')

            for i, (f_v, f_a) in enumerate(zip(f_ven_measures, f_at_measures)):
                # Read data in ventricular files,
                # extract diastolic Cai, Cai amplitude, Cai decay, and Nai
                headers_v, tmpdata_v = myplot.read_data_file(f_v)
                headers_a, tmpdata_a = myplot.read_data_file(f_a)

                _freq = 1000 / int(os.path.basename(f_v).split('_')[0])
                if _freq <= 10:
                    ca_dia_v = tmpdata_v[-1][i_ca_i_dia]
                    ca_dia_a = tmpdata_a[-1][i_ca_i_dia]
                    panel_1.append([_freq, ca_dia_v, ca_dia_a])

                    ca_max_v = tmpdata_v[-1][i_ca_i_max]
                    ca_max_a = tmpdata_a[-1][i_ca_i_max]
                    panel_2.append(
                        [_freq, ca_max_a / ca_dia_a, ca_max_v / ca_dia_v])

                    panel_3.append(
                        [_freq, tmpdata_v[-1][i_ca_i_tau], tmpdata_a[-1][i_ca_i_tau]])

                    panel_4.append(
                        [_freq, tmpdata_v[-1][i_na_dia], tmpdata_a[-1][i_na_dia]])

            panel_1 = np.array(panel_1)
            panel_2 = np.array(panel_2)
            panel_3 = np.array(panel_3)
            panel_4 = np.array(panel_4)

            axes[0][0].plot(panel_1[:, 0], panel_1[:, 1], 'o-')
            axes[0][0].plot(
                panel_1[:, 0], panel_1[:, 2], 'o-', markerfacecolor='white')
            axes[0][0].set_ylabel('Diastolic Ca_i [uM]')

            axes[0][1].plot(
                panel_2[:, 0], panel_2[:, 1], 'o-', markerfacecolor='white')
            axes[0][1].plot(panel_2[:, 0], panel_2[:, 2], 'o-')
            axes[0][1].set_ylabel('Ca_i Amplitude [folds]')

            axes[1][0].plot(
                panel_3[:, 0], panel_3[:, 1], 'o-', markerfacecolor='white')
            axes[1][0].plot(panel_3[:, 0], panel_3[:, 2], 'o-')
            axes[1][0].set_ylabel('Ca_i decay tau [ms]')

            axes[1][1].plot(
                panel_4[:, 0], panel_4[:, 1], 'o-', markerfacecolor='white')
            axes[1][1].plot(panel_4[:, 0], panel_4[:, 2], 'o-')
            axes[1][1].set_ylabel('Na_i [mM]')

        elif c_num == 2 and f_num == 23:

            f_v = 'output/out_ventricle_WT/1000_currents_burst_ISO-0.dat'
            f_a = 'output/out_atria_WT/1000_currents_burst_ISO-0.dat'

            fig, axes = plt.subplots(nrows=5, ncols=2)

            headers_v, data_v = myplot.read_data_file(f_v)
            headers_a, data_a = myplot.read_data_file(f_a)

            i_v = headers_v.index('V_m')
            i_ca_i = headers_v.index('Ca_i')
            i_phos_ltcc = headers_v.index('Phos_LCC_CK')
            i_phos_ryr = headers_v.index('Phos_RyR_CK')
            i_phos_plb = headers_v.index('Phos_PLB_CK')

            axes[0][0].plot(data_v[:, 0], data_v[:, i_v])
            axes[0][1].plot(data_a[:, 0], data_a[:, i_v])

            axes[1][0].plot(data_v[:, 0], data_v[:, i_ca_i])
            axes[1][1].plot(data_a[:, 0], data_a[:, i_ca_i])

            axes[2][0].plot(data_v[:, 0], data_v[:, i_phos_ltcc])
            axes[2][1].plot(data_a[:, 0], data_a[:, i_phos_ltcc])

            axes[3][0].plot(data_v[:, 0], data_v[:, i_phos_ryr])
            axes[3][1].plot(data_a[:, 0], data_a[:, i_phos_ryr])

            axes[4][0].plot(data_v[:, 0], data_v[:, i_phos_plb])
            axes[4][1].plot(data_a[:, 0], data_a[:, i_phos_plb])

        elif c_num == 3 and f_num == 6:

            f_wt = 'output/out_atria_WT/1000_currents_ISO-0.dat'
            f_oe = 'output/out_atria_OE/1000_currents_ISO-0.dat'
            f_ko = 'output/out_atria_KO/1000_currents_ISO-0.dat'

            fig, axes = plt.subplots(nrows=3, ncols=2)

            headers_wt, data_wt = myplot.read_data_file(f_wt)
            headers_oe, data_oe = myplot.read_data_file(f_oe)
            headers_ko, data_ko = myplot.read_data_file(f_ko)

            i_t = headers_wt.index('t')
            i_v = headers_wt.index('V_m')

            axes[0][0].plot(data_wt[:, i_t], data_wt[:, i_v], label='WT')
            axes[0][0].plot(data_oe[:, i_t], data_oe[:, i_v], label='OE')
            axes[0][0].plot(data_ko[:, i_t], data_ko[:, i_v], label='KO')
            axes[0][0].set_xlim(18990, 19100)

            axes[0][0].legend()

        elif c_num == 3 and f_num == 10:
            f_wt = 'output/out_atria_WT/currents_bcl-1000_burst-100,120_ISO-0.dat'
            f_ko = 'output/out_atria_KO/currents_bcl-1000_burst-100,120_ISO-0.dat'

            fig, axes = plt.subplots(nrows=8, ncols=2)

            headers_wt, data_wt = myplot.read_data_file(f_wt)
            headers_ko, data_ko = myplot.read_data_file(f_ko)

            i_t = headers_wt.index('t')
            i_v = headers_wt.index('V_m')
            i_na_i = headers_wt.index('Na_myo')
            i_ca_i = headers_wt.index('Ca_i')
            i_ca_sr = headers_wt.index('Ca_sr')
            i_phos_ltcc = headers_wt.index('Phos_LCC_CK')
            i_phos_ryr = headers_wt.index('Phos_RyR_CK')
            i_phos_plb = headers_wt.index('Phos_PLB_CK')

            axes[0][0].plot(data_wt[:, i_t], data_wt[:, i_v], label='WT')
            axes[0][0].set_title('WT')
            axes[0][1].plot(data_ko[:, i_t], data_ko[:, i_v], label='KO')
            axes[0][1].set_title('KO')

            axes[1][0].plot(data_wt[:, i_t], data_wt[:, i_na_i], label='WT')
            axes[1][1].plot(data_ko[:, i_t], data_ko[:, i_na_i], label='KO')

            axes[2][0].plot(data_wt[:, i_t], data_wt[:, i_ca_i], label='WT')
            axes[2][1].plot(data_ko[:, i_t], data_ko[:, i_ca_i], label='KO')

            axes[3][0].plot(data_wt[:, i_t], data_wt[:, i_ca_sr], label='WT')
            axes[3][1].plot(data_ko[:, i_t], data_ko[:, i_ca_sr], label='KO')

            axes[5][0].plot(data_wt[:, i_t], data_wt[:, i_phos_ltcc], label='WT')
            axes[5][1].plot(data_ko[:, i_t], data_ko[:, i_phos_ltcc], label='KO')

            axes[6][0].plot(data_wt[:, i_t], data_wt[:, i_phos_ryr], label='WT')
            axes[6][1].plot(data_ko[:, i_t], data_ko[:, i_phos_ryr], label='KO')

            axes[7][0].plot(data_wt[:, i_t], data_wt[:, i_phos_plb], label='WT')
            axes[7][1].plot(data_ko[:, i_t], data_ko[:, i_phos_plb], label='KO')

        elif c_num == 3 and f_num == 11:
            f_wt = 'output/out_atria_WT/currents_bcl-1000_burst-100,120_ISO-0.dat'
            f_oe = 'output/out_atria_OE/currents_bcl-1000_burst-100,120_ISO-0.dat'

            fig, axes = plt.subplots(nrows=8, ncols=2)

            headers_wt, data_wt = myplot.read_data_file(f_wt)
            headers_oe, data_oe = myplot.read_data_file(f_oe)

            i_t = headers_wt.index('t')
            want = ['V_m', 'Na_myo', 'Ca_i', 'Ca_sr',
                    'Phos_LCC_CK', 'Phos_RyR_CK', 'Phos_PLB_CK']
            i_want = list(map(headers_wt.index, want))

            for i in range(len(want)):
                axes[i][0].plot(data_wt[:, i_t], data_wt[:, i_want[i]], label='WT')
                axes[i][1].plot(data_oe[:, i_t], data_oe[:, i_want[i]], label='KO')

            axes[0][0].set_title('WT')
            axes[0][1].set_title('KO')

        # Fig2, AP, currents, and AP characteristics #
        elif f_num_inpaper == 2:
            mpl.rcParams['font.size'] = 8
            mpl.rcParams['lines.linewidth'] = 1.5

            mpl.rcParams['xtick.labelsize'] = 8
            mpl.rcParams['xtick.major.pad'] = 6
            mpl.rcParams['xtick.major.width'] = 1.5
            mpl.rcParams['ytick.labelsize'] = 8
            mpl.rcParams['ytick.major.pad'] = 6
            mpl.rcParams['ytick.major.width'] = 1.5

            mpl.rcParams['axes.labelsize'] = 8
            mpl.rcParams['axes.linewidth'] = 1.5
            mpl.rcParams['axes.unicode_minus'] = False
            mpl.rcParams['axes.labelweight'] = 'normal'

            mpl.rcParams['legend.fontsize'] = 8

            fig, axes = plt.subplots(10, 2, sharex='all', figsize=(6, 8.75),
                                     gridspec_kw={'left': 0.2})
            figures.append(fig)

            str_atr_data = "output/out_atria_WT/currents_bcl-1000_ISO-0.dat"
            str_ven_data = "output/out_ventricle_WT/currents_bcl-1000_ISO-0.dat"

            headers_atr, _data_atr = myplot.read_data_file(str_atr_data)
            headers_ven, _data_ven = myplot.read_data_file(str_ven_data)

            data_atr = _data_atr[-10001:]
            data_ven = _data_ven[-10001:]

            i_t = headers_atr.index('t')
            data_atr[:, i_t] -= data_atr[0, i_t]
            data_ven[:, i_t] -= data_ven[0, i_t]

            want = ['V_m', 'I_Na', 'I_CaL', 'I_to', 'I_Kur', 'I_Kss',
                    'I_Kr', 'I_K1', 'I_NCX', 'I_Nak']
            i_want = list(map(headers_atr.index, want))

            for i in range(len(want)):
                axes[i][0].plot(data_atr[:, i_t], data_atr[:, i_want[i]], "-k", label='Atrium')
                axes[i][1].plot(data_ven[:, i_t], data_ven[:, i_want[i]], "--k", label='Ventricle')

                axes[i][0].locator_params(axis='y', tight=True, nbins=3)

                axes[i][0].tick_params(direction='out')
                axes[i][1].tick_params(direction='out')
                if i < len(want) - 1:
                    axes[i][0].get_xaxis().set_visible(False)
                    axes[i][1].get_xaxis().set_visible(False)
                    axes[i][0].spines['bottom'].set_visible(False)
                    axes[i][1].spines['bottom'].set_visible(False)

            axes[0][0].set_xlim([0, 100])
            axes[0][1].legend()

        # Fig 10, Ca2+ handling #
        elif f_num_inpaper == '10':
            # FigA, Ca amplitude #
            # __fig_size = [i / 4 for i in _fig_size]
            # figa = plt.figure(figsize=__fig_size)
            # figures.append(figa)
            # axe = plt.subplot()
            #
            # data_amp = [[2.25, 0, "Model"], [2.4, 0.2, "Xie et al."], [2.38, 0.1, 'Li et al.']]
            # # _width = 0.2  # width of a bar
            # x_pos, _width = bar_arrange(len(data_amp))
            # for i, _data in enumerate(data_amp):
            #     if data_amp[i][2] == 'Model':  # has a revert color scheme
            #         axe.bar(x_pos[0][i], data_amp[i][0],
            #                 align='center', width=_width, color='k', linewidth=2)
            #         axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
            #                      rotation='vertical', va='bottom', ha='center', color='w')
            #     else:
            #         axe.bar(x_pos[0][i], data_amp[i][0],
            #                 align='center', width=_width, color='w', linewidth=2)
            #         axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
            #                      rotation='vertical', va='bottom', ha='center')
            #     axe.errorbar(x_pos[0][i], data_amp[i][0], yerr=data_amp[i][1], color='k', capthick=1.5)
            #
            # axe.set_xlim([0, 1])
            # axe.set_xticks([])
            # axe.set_ylabel("Amplitude (folds)")
            # figa.tight_layout()
            #
            # # FigB, Ca time to peak #
            # __fig_size = [i / 4 for i in _fig_size]
            # figb = plt.figure(figsize=(__fig_size[0] * 1.18, __fig_size[1]))
            # figures.append(figb)
            # axe = plt.subplot()
            #
            # data_amp = [[22.2, 0, "Model"], [22.3, 1, "Mancarella et al."],
            #             [22, 0, 'Li et al.'], [26, 0, 'Escobar et al.']]
            # # _width = 0.2  # width of a bar
            # x_pos, _width = bar_arrange(len(data_amp))
            # for i, _data in enumerate(data_amp):
            #     if data_amp[i][2] == 'Model':  # has a revert color scheme
            #         axe.bar(x_pos[0][i], data_amp[i][0],
            #                 align='center', width=_width, color='k', linewidth=2)
            #         axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
            #                      rotation='vertical', va='bottom', ha='center', color='w')
            #     else:
            #         axe.bar(x_pos[0][i], data_amp[i][0],
            #                 align='center', width=_width, color='w', linewidth=2)
            #         axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
            #                      rotation='vertical', va='bottom', ha='center')
            #     axe.errorbar(x_pos[0][i], data_amp[i][0], yerr=data_amp[i][1], color='k', capthick=1.5)
            #
            # axe.set_xlim([0, 1])
            # axe.set_xticks([])
            # axe.set_ylabel("Time to Peak (ms)")
            # figb.tight_layout()
            #
            # # FigC, Ca decay, tau #
            # __fig_size = [i / 4 for i in _fig_size]
            # figc = plt.figure(figsize=(__fig_size[0] * 0.85, __fig_size[1]))
            # figures.append(figc)
            # axe = plt.subplot()
            #
            # data_amp = [[111, 0, "Model"], [112, 9, "Li et al."]]
            # # _width = 0.2  # width of a bar
            # x_pos, _width = bar_arrange(len(data_amp))
            # for i, _data in enumerate(data_amp):
            #     if data_amp[i][2] == 'Model':  # has a revert color scheme
            #         axe.bar(x_pos[0][i], data_amp[i][0],
            #                 align='center', width=_width, color='k', linewidth=2)
            #         axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
            #                      rotation='vertical', va='bottom', ha='center', color='w')
            #     else:
            #         axe.bar(x_pos[0][i], data_amp[i][0],
            #                 align='center', width=_width, color='w', linewidth=2)
            #         axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
            #                      rotation='vertical', va='bottom', ha='center')
            #     axe.errorbar(x_pos[0][i], data_amp[i][0], yerr=data_amp[i][1], color='k', capthick=1.5)
            #
            # axe.set_xlim([0, 1])
            # axe.set_xticks([])
            # axe.set_ylabel("Decay " + r"$\tau$" + " (ms)")
            # figc.tight_layout()

            # FigD, Frac Ca release #
            __fig_size = [i / 4 for i in self._fig_size]
            figd = plt.figure(figsize=(__fig_size[0] * 1.43, __fig_size[1]))
            figures.append(figd)
            axe = plt.subplot()

            data_amp = [[0.485, 0, "Model"], [0.485, 0.015, 'Xie et al.'], [0.508, 0.04, 'Mancarella et al.'],
                        [0.55, 0.05, "Li et al."], [0.3, 0.01, 'Walden et al.']]
            # _width = 0.2  # width of a bar
            x_pos, _width = bar_arrange(len(data_amp))
            print(x_pos)
            for i, _data in enumerate(data_amp):
                if _data[2] == 'Model':  # has a revert color scheme
                    axe.bar(x_pos[0][i], data_amp[i][0],
                            align='center', width=_width, color='k', linewidth=2)
                    axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
                                 rotation='vertical', va='bottom', ha='center', color='w')
                else:
                    axe.bar(x_pos[0][i], data_amp[i][0],
                            align='center', width=_width, color='w', linewidth=1.5, edgecolor='k')
                    if data_amp[i][2] != 'Walden et al.':
                        axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
                                     rotation='vertical', va='bottom', ha='center')
                    else:
                        axe.annotate(data_amp[i][2], xy=(x_pos[0][i] + 0.13, 0.05), xycoords='axes fraction',
                                     rotation='vertical', va='bottom', ha='center')
                    axe.errorbar(x_pos[0][i], data_amp[i][0], yerr=data_amp[i][1], color='k', capthick=1.5, capsize=3)

            # Create the formatter using the function to_percent. This multiplies all the
            # default labels by 100, making them all percentages
            formatter = FuncFormatter(self.to_percent)

            # Set the formatter
            axe.yaxis.set_major_formatter(formatter)

            axe.set_xlim([0, 1])
            axe.set_xticks([])
            axe.set_ylabel("Fractional SR Ca" + r"$^{2+}$" + " Release")

            # FigF, frequency-dependent Ca traces #
            # __fig_size = [i / 2 for i in _fig_size]
            # figf = plt.figure(figsize=(__fig_size[0] / 1.2, __fig_size[1] / 2))
            # figures.append(figf)
            # axe = plt.subplot()
            #
            # bcl = [200, 500, 1000]
            # for _bcl in bcl:
            #     str_atr_data = "/Users/fairly/Documents/workspace_cpp/weijian_origin/bin/output/out_atria_WT
            # /currents_bcl" \
            #                    "-" + str(_bcl) + "_ISO-0.dat"
            #     _headers_atr, _data_atr = myplot.read_data_file(str_atr_data)
            #     _d = _data_atr[-2 * _bcl - 50:]
            #
            #     i_ca_i = _headers_atr.index("Ca_i")
            #     i_t = _headers_atr.index('t')
            #
            #     print(type(_d))
            #     print(_d[0])
            #     print(len(_headers_atr))
            #     print(len(_data_atr[0]))
            #     _d['t'] -= 28000
            #     _d['t'] = _d['t'] - 2000 + 2 * _bcl
            #
            #     axe.plot(_d['t'], _d["Ca_i"], 'k', label="BCL " + str(_bcl))
            #
            #     axe.set_xlim([-20, 400])
            #     axe.set_xlabel("Time (ms)")
            #     axe.set_ylabel(r"[Ca]$_{i}$")
            #     axe.legend()

        # FigE, Ca traces with Caffeine #
        elif f_num_inpaper == 101:  # number is 101 because this fig costs long time to plot. do it alone
            __fig_size = [i / 2 for i in self._fig_size]
            fige = plt.figure(figsize=(__fig_size[0], __fig_size[1] / 1.8))
            figures.append(fige)
            axe = plt.subplot()

            str_atr_data = "output/out_atria_WT/caffeine_currents_bcl-2000_burst-5000,1_ISO-0.dat"
            _headers_atr, _data_atr = myplot.read_data_file(str_atr_data)

            i_ca_i = _headers_atr.index("Ca_i")
            i_t = _headers_atr.index('t')
            _data_t = _data_atr[:, i_t]
            _data_t /= 1000
            _data_t -= 14
            # _data_cai = _data_atr[:, i_ca_i]

            axe.plot(_data_atr[:, 0], _data_atr[:, i_ca_i], color='k')

            axe.set_xlim([-0.4, 20])
            axe.set_ylabel(r"[Ca]$_{i}\ (\mu M)$")
            axe.set_xlabel('Time (s)')
            fige.tight_layout()

        else:
            if c_num is not None:
                print('Figure %s.%s has not been reproduced. Choose another figures please.'
                      % (c_num, f_num))
            else:
                print('Figure %s has not been reproduced. Choose another figure please.'
                      % f_num_inpaper)
            return

        # do some post processing, these properties cannot be set in rcParam
        for _fig in figures:
            _fig.tight_layout()

            for _axe in _fig.axes:
                _axe.spines['right'].set_visible(False)
                _axe.spines['top'].set_visible(False)
                _axe.yaxis.set_ticks_position('left')
                _axe.xaxis.set_ticks_position('bottom')
                _axe.tick_params(direction='out')

        # save figures or just show up
        suffix = '.eps'
        if args['--save']:
            outfolder = 'output/fig/'
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

            for i, fig in enumerate(figures):
                if c_num is not None:
                    figname = 'Figure' + str(c_num) + '_' + str(f_num) + '_' + chr(ord('A') + i) + suffix
                else:
                    figname = 'Figure' + str(f_num_inpaper) + '_' + chr(ord('A') + i) + suffix
                fig.savefig(os.path.join(outfolder, figname))
        else:
            plt.show()


class con3d(AbstractCommand):
    """
usage:  con3d [options] <DIR>

All images should have the same dimension.

Options:
  -b          Output binary vtk file, otherwise ASCII. Binary always faster.
  -m=MAX      Max num of images to process.
  -n=START    Set the starting number of image in the processing.
  -s=suffix   The suffix of the filenames of input images. [default: .tif]
  -o=fname    Output file name without suffix. If '-p' is provided, this fname
              defined the output path. [default: 3D_reconstruction]
  -t=dtype    Specify the underlying type of np.ndarray representing images.
              Support '?'(bool), 'u1'(uint8), 'u2'(uint16), 'f8'(float64).
              If not specified, output as input.
  -r=ratio    Ratio of the resolution of z-axis and xy-axis. [default: 1.0]
  -p          Change a perspective and save images.
  -i          Change intensity of all points for auto-contrast change.

Arguments:
  <DIR>       The directory containing images.
    """

    def execute(self):
        schema = Schema({
            '-b': bool,
            '-m': Or(None, Use(int)),
            '-n': Or(None, Use(int)),
            '-s': And(str, len),
            '-o': And(str, len),
            '-t': Or(None, Use(np.dtype)),
            '-r': Use(float),
            '-p': bool,
            '-i': bool,
            '<DIR>': Or(None, os.path.isdir),
        })

        args = schema.validate(self.args)

        files = [os.path.join(args['<DIR>'], f)
                 for f in os.listdir(args['<DIR>'])
                 if os.path.isfile(os.path.join(args['<DIR>'], f))
                 and f.endswith(args['-s'])]

        if args['-m']:
            num_of_figure = min(args['-m'], len(files))
        else:
            num_of_figure = len(files)

        # Read one of the figures to get its dimensions
        filename = files[0]
        image_ori = imread(filename)

        y, x = image_ori.shape  # Get the dimensions of figures

        if args['-t'] is not None:
            ndtype = np.dtype(args['-t'])
        else:
            ndtype = image_ori.dtype

        # choose the right function to convert the data type in an image
        map_ndtype2func = {
            np.bool_: skimage.img_as_bool,
            np.uint8: skimage.img_as_ubyte,
            np.uint16: skimage.img_as_uint,
            np.float64: skimage.img_as_float,
        }
        convert_func = map_ndtype2func[ndtype.type]

        # initialise variables
        volume = np.zeros((num_of_figure, y, x), dtype=ndtype)
        if args['-n'] and args['-n'] + num_of_figure <= len(files):
            start = args['-n']
        else:
            start = 0

        begin = time.clock()
        # Iterate all figures to fill the 3D volume
        for i in range(0, num_of_figure):
            filename = files[start + i]

            print('Reading image = %s' % filename)
            image_ori = imread(filename)

            image_ori = convert_func(image_ori)

            volume[i, :, :] = image_ori

        if args['-p']:
            if not os.path.exists(args['-o']):
                os.mkdir(args['-o'])

            outputfiles = [os.path.join(args['-o'], str(i) + '.tif') for i in range(y)]

            for i in range(y):
                imsave(outputfiles[i], volume[:, i, :])

            exit(0)

        if args['-i']:
            t = volume.dtype
            volume = volume * (np.iinfo(volume.dtype).max / volume.max())
            volume = volume.astype(t)

        print()
        print('Total image processing time: %f s.' % (time.clock() - begin))
        print('Single image processing time: %f s.' % ((time.clock() - begin) / num_of_figure))
        print()

        begin = time.clock()
        print('Constructing vtk file ...')
        write_scalar_vtk(volume, args['-r'], args['-o'], ifbinary=args['-b'])
        print('Total vtk output time: %f s.' % (time.clock() - begin))
        print('Single slice output time: %f s.' % ((time.clock() - begin) / num_of_figure))


class prep(AbstractCommand):
    """
usage:  prep [options] <PATH>

Options:
  -c             Crop void areas.
  -d=rate        Do down-sampling. `rate` should be a float number.
  -m=MAX         Max num of images to process.
  -n=nth         Precess only the nth image.
  -s=suffix      The suffix of the file names of input images. [default: .tif]
  -o=dir_name    Output dir name. [default: down_sample]
  -p             Plot intermediate images for debug purposes instead of saving result images.
  -i             Whether improve the images. (only de-noise now)

Arguments:
  PATH           The directory containing images or a single image file.
    """

    def execute(self):
        schema = Schema({
            '-c': bool,
            '-d': Or(None, Use(float)),
            '-m': Or(None, Use(int)),
            '-n': Or(None, Use(int)),
            '-s': And(str, len),
            '-o': And(str, len),
            '-p': bool,
            '-i': bool,
            '<PATH>': Or(os.path.isdir, os.path.isfile),
        })

        args = schema.validate(self.args)

        import image_preprocessing
        image_preprocessing.do(args)
