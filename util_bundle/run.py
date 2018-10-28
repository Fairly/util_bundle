#!/usr/bin/env python3
"""
Usage:
    run.py <command> [<args>...]
    run.py -h

Command:
  vclean        Refine data for voltage-clamp results.
  clean         Clean a result file for plotting.
  datareduce    A quick method reducing data from various data files.
  align         Align data in results for reasonable comparison.
  measure       Calculate characteristics of APs from results of cell models.

  eplot         Very easy plot with limited functionality.
  qplot         Plot AP or current traces by a simple script.
  plot          Plot figures.
  plotvc        Plot the curve of voltage clamp protocol.

  equa          Generate Latex equations from C code.

  con3d         Do 3D geometry construction.
  prep          Pre-process original images.

Options:
  -h, --help    Print the help message.

Arguments:
  <args>        Arguments for commands.
"""

from __future__ import division, print_function

from schema import Schema
from docopt import docopt, DocoptExit

import commands


def main():
    args = docopt(__doc__, options_first=True)

    schema = Schema({
        '--help': bool,
        '<command>': str,
        '<args>': list,
    }
    )

    args = schema.validate(args)

    # For debug
    # print(args)

    # Retrieve the command to execute.
    command_name = args.pop('<command>')

    # Retrieve the command arguments.
    command_args = args.pop('<args>')
    if command_args is None:
        command_args = {}

    # After 'poping' '<command>' and '<args>', what is left in the args dictionary are the global arguments.

    # Retrieve the class from the 'commands' module.
    try:
        command_class = getattr(commands, command_name)
    except AttributeError:
        print('Unknown command. RTFM!.')
        raise DocoptExit()

    # Create an instance of the command.
    command = command_class(command_args, args)

    # Execute the command.
    command.execute()


if __name__ == '__main__':
    main()
