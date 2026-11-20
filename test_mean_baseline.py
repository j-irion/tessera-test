import tempfile
import unittest
from pathlib import Path

import numpy as np

from mean_baseline import evaluate


class MeanBaselineTest(unittest.TestCase):
    def test_ignores_masked_and_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.npz"
            np.savez(path, target=np.array([[1., np.nan], [3., 999.]]),
                     valid=np.array([[True, True], [True, False]]))
            result = evaluate([path], 2.)
            self.assertEqual(result["labels"], 2)
            self.assertEqual(result["rmse_mg_ha"], 1.)
            self.assertEqual(result["r2"], 0.)


if __name__ == "__main__":
    unittest.main()
