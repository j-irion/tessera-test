import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_enmap import input_stats, save_input


class EnmapPreparationTest(unittest.TestCase):
    def test_statistics_ignore_nonfinite(self):
        values = np.array([[[1., np.nan], [3., np.inf]], [[2., 4.], [6., 8.]]], np.float32)
        sums, squares, counts = input_stats(values)
        np.testing.assert_array_equal(sums, [4, 20])
        np.testing.assert_array_equal(squares, [10, 120])
        np.testing.assert_array_equal(counts, [2, 4])

    def test_integer_conversion_is_lossless_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.npz"
            values = np.array([[[0., 12.], [100., 65535.]]], np.float32)
            save_input(path, values)
            with np.load(path) as data:
                self.assertEqual(data["image"].dtype, np.uint16)
                np.testing.assert_array_equal(data["image"], values)
            with self.assertRaises(FileExistsError):
                save_input(path, values)

    def test_fractional_values_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.npz"
            values = np.array([[[1.25, np.nan]]], np.float32)
            save_input(path, values)
            with np.load(path) as data:
                self.assertEqual(data["image"].dtype, np.float32)
                np.testing.assert_array_equal(data["image"], values)


if __name__ == "__main__":
    unittest.main()
