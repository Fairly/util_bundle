import unittest
from pprint import pprint

import numpy as np

from util_bundle.plot import get_axes_num, iflastrow
from util_bundle.util import bar_arrange


class TestUtilMethods(unittest.TestCase):
    def test_bar_arrange(self):
        x_pos, width = bar_arrange(2)
        gold_standard = [[0.3333333333333333, 0.6666666666666667]]
        self.assertAlmostEqual(x_pos, gold_standard)
        self.assertAlmostEqual(width, 0.2222222222222222)

        x_pos, width = bar_arrange(5)
        gold_standard = [[0.16666666666666666, 0.33333333333333337, 0.5, 0.6666666666666667, 0.8333333333333335]]
        self.assertAlmostEqual(x_pos, gold_standard)
        self.assertAlmostEqual(width, 0.1111111111111111)

        x_pos, width = bar_arrange(2, 2)
        gold_standard = [[0.2698412698412698, 0.6031746031746031], [0.3968253968253968, 0.7301587301587301]]
        self.assertAlmostEqual(x_pos, gold_standard)
        self.assertAlmostEqual(width, 0.09523809523809523)

        x_pos, width = bar_arrange(5, 3)
        gold_standard = [
            [0.12626262626262627, 0.292929292929293, 0.45959595959595967, 0.6262626262626264, 0.7929292929292932],
            [0.16666666666666669, 0.33333333333333337, 0.5000000000000001, 0.6666666666666667, 0.8333333333333335],
            [0.2070707070707071, 0.3737373737373738, 0.5404040404040404, 0.7070707070707071, 0.8737373737373738]]
        self.assertAlmostEqual(x_pos, gold_standard)
        self.assertAlmostEqual(width, 0.030303030303030304)


class TestPlotMethods(unittest.TestCase):
    def test_get_axes_num(self):
        self.assertEqual(get_axes_num(10), (4, 3))
        self.assertEqual(get_axes_num(8), (4, 2))
        self.assertEqual(get_axes_num(6), (3, 2))
        self.assertEqual(get_axes_num(7), (4, 2))
        self.assertEqual(get_axes_num(9), (3, 3))
        self.assertEqual(get_axes_num(3), (3, 1))

    def test_iflastrow(self):
        self.assertTrue(iflastrow(3, 10, 12))
        self.assertTrue(iflastrow(3, 11, 12))
        self.assertTrue(iflastrow(3, 12, 12))
        self.assertTrue(iflastrow(3, 9, 11))
        self.assertTrue(iflastrow(2, 6, 7))
        self.assertTrue(iflastrow(1, 3, 3))
        self.assertFalse(iflastrow(1, 2, 3))
        self.assertFalse(iflastrow(3, 9, 12))
        self.assertFalse(iflastrow(3, 6, 12))
        self.assertFalse(iflastrow(3, 3, 10))
        self.assertFalse(iflastrow(3, 8, 12))
        self.assertFalse(iflastrow(3, 7, 12))

if __name__ == '__main__':
    x = np.array([(1.5, 2.5, (1.0, 2.0)), (3., 4., (4., 5.)), (1., 3., (2., 6.))],
                 dtype=[('x', 'f4'), ('y', np.float32), ('value', 'f4', (2, 2))])
    pprint(x['x'])
