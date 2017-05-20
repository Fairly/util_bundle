import matplotlib as mpl


def set_matplotlib_default():
    # Matplotlib Style Sheet #
    # print mpl.rcParams.keys()
    fontsize = 6

    mpl.rcParams['font.family'] = 'Arial'
    # mpl.rcParams['font.serif'] = 'Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman'
    # mpl.rcParams['font.sans-serif'] = 'Avant Garde'
    # mpl.rcParams['font.cursive'] = 'Zapf Chancery'
    # mpl.rcParams['font.monospace'] = 'Courier, Computer Modern Typewriter'
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['font.weight'] = 'bold'

    # mpl.rcParams['text.usetex'] = True

    mpl.rcParams['mathtext.default'] = 'regular'

    mpl.rcParams['figure.figsize'] = 5.5, 4
    mpl.rcParams['figure.dpi'] = 300
    # mpl.rcParams['figure.subplot.bottom'] = 0.18

    mpl.rcParams['axes.labelsize'] = fontsize
    # mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['lines.dash_capstyle'] = 'round'
    mpl.rcParams['lines.solid_capstyle'] = 'round'

    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['xtick.major.size'] = fontsize / 2.5
    mpl.rcParams['xtick.major.pad'] = fontsize / 2.5
    mpl.rcParams['xtick.major.width'] = fontsize / 6.0
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['ytick.major.size'] = fontsize / 2.5
    mpl.rcParams['ytick.major.pad'] = fontsize / 2.5
    mpl.rcParams['ytick.major.width'] = fontsize / 6.0

    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.fontsize'] = fontsize - 1
    mpl.rcParams['legend.loc'] = 'best'
    mpl.rcParams['legend.handlelength'] = fontsize / 3.0

    mpl.rcParams['errorbar.capsize'] = fontsize / 4.0


point = 1 / 72  # inch

# Preferences for Plos Computational Biology
fig_size = (7.5, 8.75)

if __name__ == '__main__':
    print(mpl.rcParams.keys())
