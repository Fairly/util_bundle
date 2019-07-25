import os
import shutil
import sys
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from subprocess import call
import re

from docopt import docopt
from schema import Schema, Or, Use, And
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from yaml import load

from util import read_data_file, save_data_file, bar_arrange
from util_bundle.measurement import dt_recognition
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

        # deal with wildcards in Windows
        if '<FILE>' in self.args and self.args['<FILE>'] \
                and ('*' in self.args['<FILE>'][0]
                     or '?' in self.args['<FILE>'][0]
                     or '[' in self.args['<FILE>'][0]
                     or '{' in self.args['<FILE>'][0]):
            from glob import glob
            self.args['<FILE>'] = glob(self.args['<FILE>'][0])

    def execute(self):
        """Execute the commands"""
        raise NotImplementedError


class measure(AbstractCommand):
    """
usage:  measure    [options] [-t=celltype] (([-s=SUFFIX] -d=DIR) | <FILE>...)

Options:
    -m          Open multi-processing.
    --drop-last
                In tissue simulation, the last beat may not be complete. If this option is set,
                the last line will be ignored, no matter complete or not. [default: false]
    -t=celltype
                Set the cell type you are measuring to 'p' for pacemaker or 'n' for non-pacemaker.
                [default: n]
    -d=DIR      Process all files written by cell models under DIR.
    -s=SUFFIX   Set with -d=DIR to specify the suffix of file to be processed. [default: .dat]
Arguments:
    <FILE>...   Files to be processed.
    """

    def execute(self):
        schema = Schema({
            '-d': Or(None, os.path.isdir),
            '-s': Use(str),
            '-m': bool,
            '--drop-last': bool,
            '-t': Use(str),
            '<FILE>': Or(None, [os.path.isfile], error='Cannot find file[s].'),
        }
        )

        args = schema.validate(self.args)

        if args['-d']:
            # given a directory, measure all files in it
            onlyfiles = [os.path.join(args['-d'], f)
                         for f in os.listdir(args['-d'])
                         if os.path.isfile(os.path.join(args['-d'], f))]
            current_files = [f for f in onlyfiles if f.endswith(args['-s'])]
        else:
            current_files = args['<FILE>']

        process_num = cpu_count()-1 if args['-m'] else 1
        if process_num == 1:  # this single thread part is redundant but not removed for easier debugging
            for f in current_files:
                measurement.measure(f, args['-t'], args['--drop-last'])
        else:
            with ProcessPoolExecutor(process_num) as executor:
                future_list = [executor.submit(measurement.measure, f, args['-t'], args['--drop-last'])
                               for f in current_files]
                for _ in concurrent.futures.as_completed(future_list):
                    continue


def clean_result_for_plot(filename, add_underline=False, truncate_to=None, shrink=None,
                          reset_start_time=False, tail=True):  # TODO parameter `tail`
    """
    When plotting a result, it's common to reduce the size of the result file first.

    :param reset_start_time: (bool, float)
    """
    from subprocess import call
    backup_file_name = filename + '.backup.dat'
    tmp_file_name = filename + 'tmp.dat'

    print('Processing: ' + filename, end='; ', flush=True)

    shutil.copyfile(filename, backup_file_name)  # backup
    shutil.move(filename, tmp_file_name)

    if truncate_to is not None:
        f = open(tmp_file_name, 'r')
        i = len(f.readlines())
        f.close()

        if truncate_to >= i - 1:  # header in the first line, so minus 1 here
            pass
        else:
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
        multiplier = int(round(shrink / dt))

        if multiplier - 1.0 < 1e-6:
            pass
        else:
            to_file = open(filename, mode="w")
            import platform
            if platform.system() == 'Windows':
                for i, line in enumerate(open(tmp_file_name, 'r')):
                    if i == 0 or (i - 1) % multiplier == 0:
                        print(line.strip(), file=to_file)
            else:
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

    if reset_start_time is not None:
        headers, data = read_data_file(tmp_file_name)
        if data[0][0] == reset_start_time:  # don't need to reset
            pass
        else:
            time_offset = reset_start_time - data[0][0]
            data['t'] = data['t'] + time_offset
            save_data_file(filename, header=headers, data=data)

            shutil.move(filename, tmp_file_name)

    # all operations done
    shutil.move(tmp_file_name, filename)
    print('Done!')


class equa(AbstractCommand):
    """
usage: equa [options] <FILE>...

Options:
    """
    re_fnum = r'(-?\d+(\.\d+)?)([Ee]-?\d+(\.\d+)?)?'  # floating point number in C

    @staticmethod
    def find_paired_parenthesis(s):
        """
        Return the position of the paired parenthesis.
        :param s: a string starting with '(' and has its matching ')'.
        """
        rs = list(s)
        nest_tier = 0
        pos = 0
        for i in range(1, len(rs)):
            if rs[i] == '(':
                nest_tier += 1
            elif rs[i] == ')':
                if nest_tier > 0:
                    nest_tier -= 1
                else:
                    pos = i
                    break
        return pos

    @staticmethod
    def changebrace(s):
        """
        Change parentheses operators in C to braces in Latex.
        :param s: a string starting with '(' and has its matching ')'.
        """
        rs = list(s)
        rs[0] = '{'
        nest_tier = 0
        for i in range(1, len(rs)):
            if rs[i] == '(':
                nest_tier += 1
            elif rs[i] == ')':
                if nest_tier > 0:
                    nest_tier -= 1
                else:
                    rs[i] = '}'
                    break
        return ''.join(rs)

    @staticmethod
    def countleft(s):
        """
        Count characters of the numerator on the left of the '/' operator
        """
        rtn_len = 0
        if s[-2] == ')':
            s = s[:-1]
            nest_tier = 0
            for i in range(len(s) - 1, 0, -1):
                rtn_len += 1
                if s[i] == ')':
                    nest_tier += 1
                elif s[i] == '(':
                    nest_tier -= 1
                    if nest_tier <= 0:
                        # already found the matching parenthesis. But if this pair of parentheses
                        # means a function call, should look left further to add the length of
                        # the function name into the count.
                        i -= 1
                        while i >= 0 and re.match(r'\w', s[i]) is not None:
                            rtn_len += 1
                            i -= 1
                        break
        else:
            p = re.compile(r'(?<=[^\w.])([\w.{}]+)(?=/)')   # todo be aware, may cause problem when '-' is in the term
            m = re.search(p, s)
            rtn_len = len(m.group())
        return rtn_len

    @staticmethod
    def countright(s):
        rtn_len = 0
        if s[1] == '(':
            nest_tier = 0
            for i in range(len(s)):
                rtn_len += 1
                if s[i] == '(':
                    nest_tier += 1
                elif s[i] == ')':
                    nest_tier -= 1
                    if nest_tier <= 0:
                        break
        else:
            p = re.compile(r'(?<=/)([\w.{}\\-]+)(?=[^\w.]?)')
            m = re.search(p, s)
            rtn_len = len(m.group())
        return rtn_len

    def execute(self):
        schema = Schema({
            '<FILE>': Or(None, [os.path.isfile], error='Cannot find file[s].'),
        }
        )
        args = schema.validate(self.args)

        for filename in args['<FILE>']:
            fin = open(filename, 'r')
            fout = open(filename + '.tex', 'w')

            print(r"\documentclass{article}", file=fout)
            print(r"\usepackage[fleqn] {amsmath}", file=fout)
            print(file=fout)
            print(r"\begin {document}", file=fout)
            print(r"\begin {align*}", file=fout)

            for line in fin:
                line = line.strip()
                if line.startswith('//'):
                    continue
                if len(line) == 0:
                    print(r'\\', file=fout)
                    continue
                if line[0] == '%':
                    print(line, file=fout)
                    continue

                # extract unit
                p = re.compile(r'.*//.*\[(.*)\].*$')
                m = re.match(p, line)
                if m:
                    s_unit = r'\ [' + m.group(1) + ']'
                else:
                    s_unit = ''

                # clean
                line = re.sub(r'//.*', '', line)
                line = re.sub(r'\s*;\s*$', '', line)
                line = re.sub(r'^double ', '', line)
                line = re.sub(r'^float ', '', line)
                line = re.sub(r'^int ', '', line)
                line = line.replace(' ', '')
                line = line.replace('sh.', '')
                line = line.replace('sh->', '')
                line = line.replace('ec->', '')
                line = line.replace('ec.', '')
                line = line.replace('ph->', '')
                line = line.replace('ph.', '')

                # deal with fractions
                pos = 0
                while True:
                    start = line.find('/', pos)
                    if start == -1:
                        break

                    pos = start + 1
                    div_left = self.countleft(line[:start + 1])
                    div_right = self.countright(line[start:])

                    if div_left > 20 or div_right > 20:
                        numerator = line[start - div_left:start]
                        if numerator[0] == '(' and numerator[-1] == ')':
                            numerator = numerator[1:-1]

                        denominator = line[start + 1:start + div_right + 1]
                        if denominator[0] == '(' and denominator[-1] == ')':
                            denominator = denominator[1:-1]

                        frac = r'\frac {' + numerator + '}{' + denominator + '}'
                        line = line[:start - div_left] + frac + line[start + div_right + 1:]

                # deal with functions
                while line.find('exp') != -1:
                    start = line.find('exp')
                    if line[start + 3] == '(':
                        line = line[:start + 3] + self.changebrace(line[start + 3:])
                    line = line.replace('exp', 'e^', 1)
                while line.find('pow') != -1:
                    start = line.find('pow')
                    if line[start + 3] == '(':
                        end = self.find_paired_parenthesis(line[start + 3:]) + 4 + start
                        target = line[start:end]
                        target = target.replace('pow(', '')
                        target = re.sub(r',([\w]+)\)$', r'^{\1}', target)
                        line = line[:start] + target + line[end:]

                # add dot on heads of derivatives
                line = re.sub(r'ec_dot\.([\w]+)', r'\\dot{\1}', line)
                line = re.sub(r'ec_dot->([\w]+)', r'\\dot{\1}', line)
                line = re.sub(r'ecR->([\w]+)', r'\\dot{\1}', line)

                # deal with subscripts
                p = re.compile('(?<=[\w]_)([\w]+)(?=[\W_]?)')
                line = re.sub(p, r'{\1}', line)
                p = re.compile('(?<=_){([\w]+)_}')
                line = re.sub(p, r'{\1}_', line)
                p = re.compile('_([^{_]+)')  # avoid nested subscripts
                line = re.sub(p, r'\_\1', line)

                # deal with '*' operators, try to remove redundant '*'
                lp = re.compile(r'(?<=\W)' + self.re_fnum + r'\s*\Z')
                rp = re.compile(r'\A\s*' + self.re_fnum)
                while line.find('*') != -1:
                    pos = line.find('*')
                    lm = re.search(lp, line[:pos])  # check left, if is a C number
                    rm = re.search(rp, line[pos + 1:])  # check right, if is not a C number
                    if lm is not None and rm is None and not line[pos + 1:].startswith(r'\frac'):
                        line = line.replace('*', '\\,', 1)  # remove this redundant '*'
                    else:
                        line = line.replace('*', '\\times ', 1)

                # symbol replacement
                line = line.replace('tau', '\\tau ')
                line = line.replace('inf', '\\infty ')
                line = line.replace('alpha', '\\alpha ')
                line = line.replace('beta', '\\beta ')
                line = line.replace('gamma', '\\gamma ')
                line = line.replace('delta', '\\delta ')
                line = line.replace('mu', '\\mu ')
                line = line.replace('sigma', '\\sigma ')

                # scientific notation
                line = re.sub(r'(-?\d+(\.\d+)?)([Ee])(-?\d+(\.\d+)?)', r'\1\\times 10^{\4}', line)

                if line.startswith('if'):
                    line = line.replace('if', '')
                    line = line.replace(r'{', '')
                    line = line.replace('\n', '')
                    print('IF $', line, '$', file=fout)
                elif line == '}':
                    print(r'ENDIF \\', file=fout)
                else:
                    print(r'&' + line + s_unit + r'\\', file=fout)

            print("\end{align*}", file=fout)
            print("\end{document}", file=fout)
            fin.close()
            fout.close()


class align(AbstractCommand):
    """
usage: align [options] (-b=head | -t=tail) <FILE>...

Align APs in several result files to allow reasonable comparison.

Options:
    -m          Open multi-processing.
    -b=head     Align to the `head`th AP from the beginning. Starts from 1.
    -t=tail     Align to the `tail`th AP from the ending. Starts from 1.

    -o=offset   Time retained before the first aligned AP. [default: 100]
    -s=shrink   Shrink the data size as described in `clean` command.
    """

    def execute(self):
        schema = Schema({
            '-m': bool,
            '-b': Or(None, And(Use(int), lambda n: n > 0)),
            '-t': Or(None, And(Use(int), lambda n: n > 0)),
            '-o': Use(float),
            '-s': Or(None, Use(float)),
            '<FILE>': Or(None, [os.path.isfile], error='Cannot find file[s].'),
        }
        )
        args = schema.validate(self.args)

        p = ThreadPoolExecutor(cpu_count()) if args['-m'] else ThreadPoolExecutor(1)

        for fname in args['<FILE>']:
            p.submit(self.thread_align, args, fname)

    @staticmethod
    def thread_align(args, fname):
        print('Processing {}.'.format(fname))
        l_labels, npa_data = read_data_file(fname)
        APs = measurement.ap_recognition(npa_data)
        dt = measurement.dt_recognition(npa_data)
        truncate_num = 100000000000
        if args['-b']:
            truncate_num = len(npa_data) - int((APs[args['-b']][0] - args['-o']) / dt)
        elif args['-t']:
            truncate_num = len(npa_data) - int((APs[-args['-t']][0] - args['-o']) / dt)
        if truncate_num > len(npa_data):
            pass
        else:
            if args['-s']:
                clean_result_for_plot(fname, truncate_to=truncate_num, shrink=args['-s'],
                                      reset_start_time=0, tail=True)
            else:
                clean_result_for_plot(fname, truncate_to=truncate_num,
                                      reset_start_time=0, tail=True)


class clean(AbstractCommand):
    """
usage: clean rmbackup <DIR>
       clean -R <FILE>...
       clean [options] <FILE>...

Clean the data file. Original file will be backed up.

Commands:
    rmbackup    Recursively remove all backup data files in a given <DIR>.

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
    <DIR>              A directory.
    """

    def remove_backup(self, path):
        for name in os.listdir(path):
            if os.path.isdir(name):
                self.remove_backup(name)
            else:
                if name.endswith('backup.dat'):
                    os.remove(os.path.join(path, name))

    def execute(self):
        schema = Schema({'rmbackup': bool,
                         '--reset-time': Or(None, Use(float)),
                         '--truncate': Or(None, Use(int)),
                         '--shrink': Or(None, Use(float)),
                         '--underline': bool,
                         '--recover': bool,
                         '<FILE>': Or(None, list),
                         '<DIR>': Or(None, os.path.isdir),
                         }
                        )

        args = schema.validate(self.args)

        if args['--recover']:
            for f in args['<FILE>']:
                original_file = f.replace('.backup.dat', '')
                shutil.move(f, original_file)
        elif args['rmbackup']:
            self.remove_backup(args['<DIR>'])
        else:
            for f in args['<FILE>']:
                clean_result_for_plot(f, add_underline=args['--underline'],
                                      truncate_to=args['--truncate'],
                                      shrink=args['--shrink'],
                                      reset_start_time=args['--reset-time'])


class datareduce(AbstractCommand):
    """
usage: datareduce [-s=num] [-y=yaxis] [-m=mode] [options] <FILE>...

Options:
    -s=num      An integer define which part will be extracted from the file name
                as the data labels. For example, if file name is 'result_s1s2-25.dat',
                and this parameter is set to 1, 's1s2-25' will be extracted and used
                as the first column in the output. [default: 1]
    -y=yaxis    A ',' separate string specifying the column(s) used for the y-axis.
                'ALL' means reduce all columns. [default: ALL]
    -m=mode     Mode can be `t` for tail, `l` for largest, or `s` for smallest. This
                parameter sets the behaviour of extracting which value out from the
                yaxis. So it can extract the last, biggest, or smallest value. A number
                can follow the `mode` to extract more than 1 values from the file. [default: t]
    --sort-alternans
                If alternans are in the result files, more than 1 values will be extracted from
                the result files to plot the bifurcation. This option sort the values to
                avoid crossings in the output curves. [default: false]

Arguments:
    <FILE>      File names.
    """

    def execute(self):
        schema = Schema({
            '-s': Use(int),
            '-y': str,
            '-m': str,
            '--sort-alternans': bool,
            '<FILE>': [os.path.isfile],
        })

        args = schema.validate(self.args)

        first_filename = args['<FILE>'][0]
        path, name = os.path.split(first_filename)
        target = name.split('_')[args['-s']]
        xaxis_name = target[:target.find('-')]

        # number of values extracted
        n = 1
        if len(args['-m']) > 1:
            n = int(args['-m'][1:])

        if args['-y'] == 'ALL':
            yaxis_name, _ = read_data_file(first_filename)
        else:
            yaxis_name = args['-y'].split(',')

        result = []
        for fname in args['<FILE>']:
            print('Processing file: ' + fname)
            _, m = read_data_file(fname)

            if n > len(m):
                print('Number of rows in datafile is less than the number wanted to be extracted. Exit!')
                exit(1)

            target = os.path.basename(fname)
            target = os.path.splitext(target)[0]
            target = target.split('_')[args['-s']]
            xaxis = target[target.find('-') + 1:]
            xaxis = float(xaxis)

            y_result = []
            sort_flag = args['--sort-alternans']
            import heapq        # using heap sort for nth-largest and nth-smallest
            for _y_name in yaxis_name:
                if args['-m'].startswith('t'):
                    if sort_flag:
                        tmp = []
                        for i in range(n):
                            tmp.append(m[-1-i][_y_name])
                        y_result.extend(sorted(tmp))
                    else:
                        for i in range(n):
                            y_result.append(m[-1-i][_y_name])
                elif args['-m'].startswith('l'):
                    if n == 1:
                        y_result.append(max(m[_y_name]))
                    else:
                        y_result.extend(heapq.nlargest(n, m[_y_name]))
                elif args['-m'].startswith('s'):
                    if n == 1:
                        y_result.append(min(m[_y_name]))
                    else:
                        y_result.extend(heapq.nsmallest(n, m[_y_name]))
                else:
                    print('Unsupported mode "' + args['-m'] + '". Exit.', file=sys.stderr)

            result.append([xaxis, *y_result])

        # duplicate yaxis names if more than 1 value extracted
        new_yaxis = []
        if n > 1:
            for _y_name in yaxis_name:
                for i in range(n):
                    new_yaxis.append(_y_name + '_' + str(i))

        result.sort()
        if args['-y'] == 'ALL':
            outfilename = os.path.join(path, 'datareduce_ALL.dat')
        else:
            outfilename = os.path.join(path, 'datareduce_' + xaxis_name + '-' + '+'.join(yaxis_name) + '.dat')
        save_data_file(outfilename, [xaxis_name, *new_yaxis], result)


class eplot(AbstractCommand):
    """
An easy plot command for single data files. Unlike `qplot`, this command is designed to plot compact data
in each file, not extracted data from a series of files.

usage:
    eplot  [-f] [-e=func] [-L=labels] [-x=xaxis] -y=yaxis <FILE>...

Options:
    -f          New figure for each file and all `yaxis` in one figure. If not, new figure for each `yaxis`,
                and the columns named `yaxis` in every file will be on one figure.
    -e=func     A function with two arguments 'a' and 'b' that will be `eval`ed to generate results for plotting.
                For example: "1 - a/b" means define a function as: "lambda a,b: 1-a/b". The two arguments
                are actually two data fields in `yaxis`. `yaxis` will be consumed consecutively to fulfill
                the arguments and generate results for plotting. No function is allowed being called in `func`.
    -L=labels   Legends used in plotting, since these legends can either be for each file or for each `yaxis` (see
                `-f`) , this option is different to the `-L` in `qplot`. It should be a ',' separate string.
    -x=xaxis    Specify the column used for the x-axis. `xaxis` can be a number (start from 0)
                or the name of the column. [default: 0]
    -y=yaxis    A ',' separate string specifying the column(s) used for the y-axis. If set to 'all', all fields
                will be plotted.

Arguments:
    <FILE>      File names.
    """
    def execute(self):
        schema = Schema({
            '-f': bool,
            '-e': Or(None, str),
            '-x': Or(Use(int), str),
            '-y': str,
            '-L': Or(None, str),
            '<FILE>': [os.path.isfile],
        })

        args = schema.validate(self.args)

        if args['-e']:
            func = eval("lambda a, b: " + args['-e'])

        legends = []
        if args['-L']:
            legends = args['-L'].strip().split(',')

        import plot
        data = plot.gen_plot_data(args['<FILE>'])

        # re-asign the name of x-axis
        for _d in data:
            if isinstance(args['-x'], int):
                _d['xaxis'] = _d['l_field_names'][args['-x']]
            else:
                _d['xaxis'] = args['-x']

        # deal with the names of y-axis
        header = data[0]['l_field_names']
        if args['-y'] == 'all':
            l_y = header
        else:
            l_y = args['-y'].split(',')
            for i, y in enumerate(l_y):
                if y.isdigit():  # change number of columns into the header string of columns
                    l_y[i] = header[int(y)]

        # pack data into simpler structures
        xaxis_infiles = [_d['data'][_d['xaxis']] for _d in data]    # list of a column in ndarray
        yaxis_infiles = []                                          # list of list of columns in ndarray
        for j, _d in enumerate(data):
            yaxis_in_single_file = []
            for i in range(len(l_y)):
                if args['-e']:  # if self defined function
                    if len(l_y) - i < 2:
                        break
                    y0 = _d['data'][l_y[i].replace('.', '')]
                    y1 = _d['data'][l_y[i + 1].replace('.', '')]
                    y = func(y0, y1)
                    i += 2
                else:  # else, sequentially plot
                    try:
                        y = _d['data'][l_y[i].replace('.', '')]
                    except ValueError:
                        print(l_y[i], ' is not in the file. Ignore.')
                        continue
                yaxis_in_single_file.append(y)
            yaxis_infiles.append(yaxis_in_single_file)

        # plot
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*']
        index_marker = 0
        l_figure = []
        if args['-f']:
            for i, _x in enumerate(xaxis_infiles):
                l_figure.append(plt.figure())
                for j, _y in enumerate(yaxis_infiles[i]):
                    plt.plot(_x, _y, marker=markers[index_marker], label=legends[j] if legends else None)
                    index_marker += 1
        else:
            for i in range(len(yaxis_infiles[0])):
                l_figure.append(plt.figure())
                for j, _x in enumerate(xaxis_infiles):
                    plt.plot(_x, yaxis_infiles[j][i], marker=markers[index_marker], label=legends[j] if legends else None)
                    index_marker += 1

        if args['-L']:
            for _figure in l_figure:
                axes = _figure.get_axes()
                for _a in axes:
                    _a.legend()

        # save on my computer
        if os.path.isdir(r'C:\Users\Shanzhuo Zhang\Downloads'):
            print(r'Automatically save to C:\Users\Shanzhuo Zhang\Downloads\Fig_1.png')
            plt.savefig(r'C:\Users\Shanzhuo Zhang\Downloads\Fig_1.png', dpi=300)
        plt.show()


class qplot(AbstractCommand):
    """
An quick interface for plotting and comparing APs or currents.

usage:
       qplot [options] [-x=xlim -y=ylim] <FILE>...
       qplot [options] [-x=xlim -y=ylim] -L (<FILE> <LABEL>)...

Options:
    --xaxis=x   In special cases, a column used for the x-axis in plotting can be set.
                `x` is a string which is the name or number of the column used as the x-axis.

    -V          Trigger for whether plotting the AP.
    -I          Trigger for whether plotting currents.
    -A          Trigger for whether plotting all fields.
    -C=CTM      Customized plotting prefixes, separated by ','. For example,
                "V,I" means plotting fields whose name starts with "V" or "I".
                Also supports the use of '!' before a field name to remove fields from plotting.
                "!I_Ca" means not plotting fields starts with "I_Ca".
                CTM can also be ',' separated numbers. In this case, the number defines which column(s)
                to be plotted. '1,2' means the first and second column (except the real first column,
                which is usually the x-axis) will be plotted.

    -p=pnum     Max number of panels in a single figure. Set to a big value to
                plot all panels in one figure, or a small number to make the
                figure clear. [default: 12]

    -c          If set, using only black color for plotting.
    -S          Using mpl_setting.py to set default matplotlib.

    -o=OUT      The file name of the output figure.
                If not given, show the figure instead of saving it.

    -x=xlim     Set limits on the x axis. `xlim` is a comma separated string. e.g.: '10,100'
                means to plot the range [10, 100] on the x-axis.
    -y=ylim     Set limits on the y axis. Same definition with `xlim` above.

    -L          If this is set, provide labels for files and labels will be use as legends in the figure.

Arguments:
    <FILE>      File names.
    <LABEL>     One Label for one file.
    """

    def execute(self):
        schema = Schema({
            '--xaxis': Or(None, str),
            '-I': bool,
            '-V': bool,
            '-A': bool,
            '-C': Or(None, And(str, len)),
            '-p': Or(None, Use(int)),
            '-c': bool,
            '-S': bool,
            '-o': Or(None, And(str, len)),
            '-L': bool,
            '<FILE>': [os.path.isfile],
            '<LABEL>': Or(None, [str]),
            '-x': Or(None, str),
            '-y': Or(None, str),
        })

        args = schema.validate(self.args)

        header, _ = read_data_file(args['<FILE>'][0], max_rows=2)

        if header is None:
            print("Result file does not have a header. Use `eplot` instead.", file=sys.stderr)
            exit(1)

        plot_flag = []
        if args['-A']:
            plot_flag.append('all')
        else:
            if args['-V']:
                plot_flag.append('V')
            if args['-I']:
                plot_flag.append('I')
            if args['-C']:
                if args['-C'].split(',')[0].isdigit():
                    indexes = map(int, args['-C'].split(','))
                    for i in indexes:
                        plot_flag.append(header[i])
                else:
                    plot_flag.extend(args['-C'].split(','))

        xaxis = None
        if args['--xaxis'] is not None:
            if args['--xaxis'].isdigit():
                xaxis = header[int(args['--xaxis'])]
            else:
                xaxis = args['--xaxis']

        if not plot_flag:  # default: plot voltage
            plot_flag.append('V')

        plot_flag = list(set(plot_flag))  # remove duplicated items

        if args['-x']:
            xlim = [float(i) for i in args['-x'].split(',')]
        else:
            xlim = None

        if args['-y']:
            ylim = [float(i) for i in args['-y'].split(',')]
        else:
            ylim = None

        if args['-c']:
            color = 'k'
        else:
            color = None

        myplot.autoplot(args['<FILE>'], args['<LABEL>'], xaxis=xaxis,
                        flags=plot_flag, outfigname=args['-o'], xlimit=xlim, ylimit=ylim,
                        color=color, mplsetting=args['-S'], max_panel_num=args['-p'])


class plotvc(AbstractCommand):
    """
Usage:  plotvc [-n] [-s=start] [-e=end] <YAMLFILE>

Options:
  -s=start    Set the start time of plotting, since the holding time is usually too long. [default: 1]
  -e=end      Set the number in multi-seconds truncated from the end. [default: 0]

  -n          If set, no text will be plot.

Arguments:
  <YAMLFILE>  The config file containing the voltage clamp protocol.
    """

    def execute(self):
        schema = Schema({
            "-s": Use(int),
            "-e": Use(int),
            "-n": bool,
            "<YAMLFILE>": os.path.isfile,
        }
        )

        args = schema.validate(self.args)

        import plotvc as pvc
        f = open(args["<YAMLFILE>"], 'r')
        pvc.plotvc(f, start=args['-s'], end=args['-e'], iftext=not args['-n'])


class vclean(AbstractCommand):
    """
Usage:
      vclean  [options] [-s num] [(-i <START> <END>)] [-t=TARGET] -y <YAMLFILE>
      vclean  [options] [-s num] [(-i <START> <END>)] -t=TARGET <FILE>...

In the directory of voltage-clamp results, run this to get refined results.

Options:
  -i          Toggle the specification of the interested interval of data.
  -s=num      An integer define which part will be extracted from the file name as the data labels. [default: 1]
  -t=TARGET   Set the target to be extracted from results. Should be a comma separated string.
              For example, 'I_Na,I_Na_m,I_Na_h'.
  -y          Toggle the yaml model. Read all files in current dir with name prefix defined in the yaml file.

Arguments:
  <YAMLFILE>  The config file containing the voltage clamp protocol.
  <FILE>...   Result files specified.
  <START> <END>
              Starting and ending lines of data of the interested time interval in this voltage clamp.


Examples:
    File1: result_BCL-100.dat
    File2: result_BCL-1000.dat

    result_BCL-100.dat:
    t       I_Na    ...
    0.1     1
    0.2     3
    0.3     5
    ...     ...

    result_BCL-1000.dat:
    t       I_Na    ...
    0.1     10
    0.2     30
    0.3     50
    ...     ...

    If run `vclean -s 1 -y I_Na result*` ('*' is the wildcard, supported on both Linux
    and Win), the resultant file will be:

    t       BCL-100     BCL-1000
    0.1     1           10
    0.2     3           30
    0.3     5           50
    ...     ...         ...
    """

    def execute(self):
        schema = Schema({
            "-i": bool,
            "-s": Use(int),
            "-t": Or(None, str),
            "-y": bool,
            "<YAMLFILE>": Or(None, os.path.isfile),
            "<FILE>": [os.path.isfile],
            "<START>": Or(None, Use(int)),
            "<END>": Or(None, Use(int)),
        }
        )

        args = schema.validate(self.args)

        l_files = []
        if args['<YAMLFILE>']:
            yfile = open(args["<YAMLFILE>"], 'r')
            config = load(yfile)
            result_prefix = config['Config']['RESULT_FILE_NAME']

            if 'Voltage_Clamp' in config and config['Voltage_Clamp'] is not None:
                l_files = [f for f in os.listdir('.')
                           if os.path.isfile(f)
                           and f.startswith(result_prefix)]
            else:
                print('Error: No "Voltage_Clamp" is defined in the .yaml file. Exit.')
                exit(1)

            if args['-t']:
                targets = args['-t'].split(',')
            else:
                targets = [config['Voltage_Clamp']['Target']]
        else:
            l_files = args['<FILE>']
            targets = args['-t'].split(',')

        import plotvc as pvc
        pvc.vclean(l_files, targets, xaxis=args['-s'], start=args['<START>'], end=args['<END>'])


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
            data_atr = data_atr.view((data_atr.dtype[0], len(data_atr.dtype.names)))
            data_ven = data_ven.view((data_ven.dtype[0], len(data_ven.dtype.names)))
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
        elif f_num_inpaper == 10:
            # FigA, Ca amplitude #
            __fig_size = [i / 4 for i in self._fig_size]
            figa = plt.figure(figsize=__fig_size)
            figures.append(figa)
            axe = plt.subplot()

            data_amp = [[2.25, 0, "Model"], [2.4, 0.2, "Xie et al."], [5.54, 0.37, 'Li et al.'],
                        [6.33, 0.99, 'Guo et al.']]
            # _width = 0.2  # width of a bar
            x_pos, _width = bar_arrange(len(data_amp))
            for i, _data in enumerate(data_amp):
                if data_amp[i][2] == 'Model':  # has a revert color scheme
                    axe.bar(x_pos[0][i], data_amp[i][0],
                            align='center', width=_width, color='k', linewidth=2)
                    axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
                                 rotation='vertical', va='bottom', ha='center', color='w')
                else:
                    axe.bar(x_pos[0][i], data_amp[i][0],
                            align='center', width=_width, color='w', linewidth=2, edgecolor='k')
                    axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
                                 rotation='vertical', va='bottom', ha='center')
                    axe.errorbar(x_pos[0][i], data_amp[i][0], yerr=data_amp[i][1], color='k', capthick=1.5, capsize=3)

            axe.set_xlim([0, 1])
            axe.set_xticks([])
            axe.set_ylabel("Amplitude (folds)")
            figa.tight_layout()

            # FigB, Ca time to peak #
            __fig_size = [i / 4 for i in self._fig_size]
            figb = plt.figure(figsize=(__fig_size[0] * 1.18, __fig_size[1]))
            figures.append(figb)
            axe = plt.subplot()

            data_amp = [[22.2, 0, "Model"], [23.5, 2.3, "Mancarella et al."],
                        [41.8, 1.8, 'Li et al.']]  # , [26, 0, 'Escobar et al.'], this is for rat, obsolete
            # _width = 0.2  # width of a bar
            x_pos, _width = bar_arrange(len(data_amp))
            for i, _data in enumerate(data_amp):
                if data_amp[i][2] == 'Model':  # has a revert color scheme
                    axe.bar(x_pos[0][i], data_amp[i][0],
                            align='center', width=_width, color='k', linewidth=2)
                    axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
                                 rotation='vertical', va='bottom', ha='center', color='w')
                else:
                    axe.bar(x_pos[0][i], data_amp[i][0],
                            align='center', width=_width, color='w', linewidth=2, edgecolor='k')
                    axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
                                 rotation='vertical', va='bottom', ha='center')
                    axe.errorbar(x_pos[0][i], data_amp[i][0], yerr=data_amp[i][1], color='k', capthick=1.5, capsize=3)

            axe.set_xlim([0, 1])
            axe.set_xticks([])
            axe.set_ylabel("Time to Peak (ms)")
            figb.tight_layout()

            # FigC, Ca decay, tau #
            __fig_size = [i / 4 for i in self._fig_size]
            figc = plt.figure(figsize=(__fig_size[0] * 0.85, __fig_size[1]))
            figures.append(figc)
            axe = plt.subplot()

            data_amp = [[111, 0, "Model"], [215, 17, "Li et al."]]
            # _width = 0.2  # width of a bar
            x_pos, _width = bar_arrange(len(data_amp))
            for i, _data in enumerate(data_amp):
                if data_amp[i][2] == 'Model':  # has a revert color scheme
                    axe.bar(x_pos[0][i], data_amp[i][0],
                            align='center', width=_width, color='k', linewidth=2)
                    axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
                                 rotation='vertical', va='bottom', ha='center', color='w')
                else:
                    axe.bar(x_pos[0][i], data_amp[i][0],
                            align='center', width=_width, color='w', linewidth=2, edgecolor='k')
                    axe.annotate(data_amp[i][2], xy=(x_pos[0][i], 0.05), xycoords='axes fraction',
                                 rotation='vertical', va='bottom', ha='center')
                    axe.errorbar(x_pos[0][i], data_amp[i][0], yerr=data_amp[i][1], color='k', capthick=1.5, capsize=3)

            axe.set_xlim([0, 1])
            axe.set_xticks([])
            axe.set_ylabel("Decay " + r"$\tau$" + " (ms)")
            figc.tight_layout()

            # FigD, Frac Ca release #
            __fig_size = [i / 4 for i in self._fig_size]
            figd = plt.figure(figsize=(__fig_size[0] * 1.43, __fig_size[1]))
            figures.append(figd)
            axe = plt.subplot()

            data_amp = [[0.485, 0, "Model"], [0.485, 0.015, 'Xie et al.'], [0.508, 0.04, 'Mancarella et al.'],
                        [0.55, 0.05, "Li et al."]]  # , [0.3, 0.01, 'Walden et al.'] this is for rat, so obsolete
            # _width = 0.2  # width of a bar
            x_pos, _width = bar_arrange(len(data_amp))
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
            # _fig.tight_layout()

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

        import reconstruction
        reconstruction.construct_3D(args)


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


class collectvtk(AbstractCommand):
    """
usage:  collectvtk <DIR>

Arguments:
  <DIR>       The directory containing images.
    """
    def execute(self):
        schema = Schema({
            '<DIR>': os.path.isdir,
        })

        args = schema.validate(self.args)

        save_dir = os.path.join(args['<DIR>'], 'vtk_files')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        AP_dir = os.path.join(save_dir, 'AP')
        if not os.path.isdir(AP_dir):
            os.mkdir(AP_dir)

        Cai_dir = os.path.join(save_dir, 'Cai')
        if not os.path.isdir(Cai_dir):
            os.mkdir(Cai_dir)

        content = os.listdir(args['<DIR>'])
        for _name in content:
            AP_vtk = os.path.join(args['<DIR>'], _name, 'OneD.vtk')
            if os.path.isfile(AP_vtk):
                shutil.copy(AP_vtk, os.path.join(AP_dir, _name + '.vtk'))

            Cai_vtk = os.path.join(args['<DIR>'], _name, 'OneD_Cai.vtk')
            if os.path.isfile(Cai_vtk):
                shutil.copy(Cai_vtk, os.path.join(Cai_dir, _name + '.vtk'))


class cvtk(AbstractCommand):
    def __init__(self, command_args, global_args):
        cvtk.__doc__ = collectvtk.__doc__
        cvtk.execute = collectvtk.execute
        super().__init__(command_args, global_args)
