import unittest

import numpy as np

from check_coverage import tile_keys


class TileKeysTest(unittest.TestCase):
    def test_positive_and_negative_coordinates(self):
        keys = tile_keys(np.array([0.01, -0.01, -0.11]), np.array([0.01, -0.01, 0.11]))
        self.assertEqual(keys, [(5, 5), (-5, -5), (-15, 15)])


if __name__ == "__main__":
    unittest.main()
