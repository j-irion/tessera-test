import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from check_v11_exact import pixel_indices, save_new


class ExactCoverageTest(unittest.TestCase):
    def test_indices_reject_outside_grid(self):
        indices, inside = pixel_indices(np.array([5., 15., 25.]), np.array([5., 16., 100.]))
        np.testing.assert_array_equal(indices, [0, 1, 2])
        np.testing.assert_array_equal(inside, [True, True, False])

    def test_checkpoint_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.parquet"
            save_new(pd.DataFrame({"valid": [True]}), path)
            with self.assertRaises(FileExistsError):
                save_new(pd.DataFrame({"valid": [False]}), path)
            self.assertTrue(pd.read_parquet(path).valid.iloc[0])


if __name__ == "__main__":
    unittest.main()
