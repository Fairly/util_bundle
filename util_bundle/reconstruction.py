import sys
import os
import time

import numpy as np
try:
    import skimage
    from skimage.io import imread, imsave
    import skimage.filters
    import skimage.draw
except ImportError:
    print("Error: Package 'skimage' is needed when doing image processing. "
          "See 'http://scikit-image.org' for details.", file=sys.stderr)
    sys.exit(1)

from util import write_scalar_vtk


def construct_3D(args):
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
