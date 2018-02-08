import sys
import os
import time

import numpy as np
import scipy
try:
    import skimage
    from skimage.io import imread, imsave
    from skimage.transform import rescale
except ImportError:
    print("Error: Package 'skimage' is needed when doing image processing. "
          "See 'http://scikit-image.org' for details.", file=sys.stderr)
    sys.exit(1)
import matplotlib.pyplot as plt

import mpl_setting


def do(args):
    try:
        import cv2
    except ImportError:
        print('Error: Cannot import \'cv2\' from package opencv-python. Please install '
              'this package and retry.')
        return

    if os.path.isdir(args['<PATH>']):
        files = [os.path.join(args['<PATH>'], f)
                 for f in os.listdir(args['<PATH>'])
                 if os.path.isfile(os.path.join(args['<PATH>'], f))
                 and f.endswith(args['-s'])]
    else:
        files = [args['<PATH>']]

    if not os.path.exists(args['-o']):
        os.mkdir(args['-o'])

    out_files = [os.path.join(args['-o'], os.path.basename(f))
                 for f in files
                 if os.path.isfile(f)
                 and f.endswith(args['-s'])]

    if args['-m'] is not None:
        num_of_figure = min(args['-m'], len(files))
    else:
        num_of_figure = len(files)

    if args['-n'] is not None:
        files = files[args['-n']:args['-n'] + 1]
        out_files = out_files[args['-n']:args['-n'] + 1]
        num_of_figure = 1

    begin = time.clock()

    # Iterate all images
    for i in range(0, num_of_figure):
        filename = files[i]
        out_file = out_files[i]

        print('Reading image = %s' % filename)
        image_ori = imread(filename)

        if args['-i']:
            # Cut a circle from the original image
            rr, cc = skimage.draw.circle(999, 999, 800)
            img = np.zeros(image_ori.shape).astype(image_ori.dtype)
            img[rr, cc] = image_ori[rr, cc]

            # Cartesian coordinate to polar
            img = cv2.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), 800, cv2.WARP_FILL_OUTLIERS)

            # Mask construction
            mask0 = img.copy()
            mask0[mask0 < 100] = 0
            mask0[mask0 >= 100] = 1
            mask0.astype(np.bool)
            # Do one iteration of erosion
            bs = scipy.ndimage.generate_binary_structure(2, 1)
            mask0 = scipy.ndimage.binary_erosion(mask0, structure=bs, iterations=6)
            mask0 = scipy.ndimage.binary_dilation(mask0, structure=bs, iterations=6)

            img[~mask0] = 0

            # i2 = scipy.ndimage.filters.gaussian_filter(i0, 4)
            # axe = fig.add_subplot(332)
            # axe.imshow(i2)
            #
            # i3 = i0 - i2
            # i3[~mask0] = 0
            # axe = fig.add_subplot(333)
            # axe.imshow(i3)
            #
            # i4 = np.fabs(i3.min()) + i3
            # i4[~mask0] = 0
            # axe = fig.add_subplot(334)
            # axe.imshow(i4)

            # Remove ring artifacts
            i5 = scipy.ndimage.median_filter(img, size=(15, 9))
            i5[~mask0] = 0

            # Correction image
            i6 = (img - i5)
            i6[~mask0] = 0

            # Corrected polar image
            i9 = img - i6
            i9[~mask0] = 0

            # Polar coordinate to Cartesian
            i9 = cv2.linearPolar(i9, (img.shape[0] / 2, img.shape[1] / 2), 800, cv2.WARP_INVERSE_MAP)

            if args['-p']:
                mpl_setting.set_matplotlib_default()

                # original image
                plt.figure()
                plt.imshow(image_ori)

                # Show histogram of the image
                hist, bin_edges = np.histogram(img, bins=60)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                plt.figure()
                plt.subplots_adjust(left=0.2)
                plt.bar(bin_centers[1:], hist[1:], width=5)
                plt.title('Histogram before')
                plt.xlabel('Greyscale bins')
                plt.ylabel('Number of pixels')

                axe = plt.gca()
                axe.spines['right'].set_visible(False)
                axe.spines['top'].set_visible(False)
                axe.yaxis.set_ticks_position('left')
                axe.xaxis.set_ticks_position('bottom')
                axe.tick_params(direction='out')
                # plt.savefig('hist_before.jpg')

                # intermediate images
                fig = plt.figure()
                fig.tight_layout()

                axe = fig.add_subplot(339)
                axe.imshow(mask0)
                axe = fig.add_subplot(331)
                axe.imshow(img)
                axe = fig.add_subplot(335)
                axe.imshow(i5)
                axe = fig.add_subplot(336)
                axe.imshow(i6)
                axe = fig.add_subplot(337)
                axe.imshow(i9)
                axe = fig.add_subplot(338)
                axe.imshow(i9)

                # Show histogram of the image
                hist, bin_edges = np.histogram(i9, bins=60)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                plt.figure()
                plt.subplots_adjust(left=0.2)
                plt.bar(bin_centers[1:], hist[1:], width=5)
                plt.title('Histogram after')
                plt.xlabel('Greyscale bins')
                plt.ylabel('Number of pixels')

                axe = plt.gca()
                axe.spines['right'].set_visible(False)
                axe.spines['top'].set_visible(False)
                axe.yaxis.set_ticks_position('left')
                axe.xaxis.set_ticks_position('bottom')
                axe.tick_params(direction='out')
                # plt.savefig('hist_after.jpg')

                plt.show()
                continue

            image_ori = i9

        # crop background
        if args['-c']:
            image_ori = image_ori[400:1440, 550:1740].copy()
            plt.figure()
            plt.imshow(image_ori)

        # down-sampling
        if args['-d']:
            image_ori = rescale(image_ori, args['-d'], mode='reflect')

        # image_ori = rescale_intensity(image_ori)  # intensity adjustment
        if np.issubdtype(image_ori.dtype.type, np.float):
            image_ori = skimage.img_as_uint(image_ori)

        print("Saving image = %s" % out_file)
        imsave(out_file, image_ori)

    print('Total processing time: %f s.' % (time.clock() - begin))
