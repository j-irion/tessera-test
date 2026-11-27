import unittest

import torch

from evaluate_biomass_bins import bin_sums


class BinSumsTest(unittest.TestCase):
    def test_boundaries_masks_and_errors(self):
        target = torch.tensor([0., 99.9, 100., 199.9, 200., 299.9, 300., 399.9, 400., 500., float("nan")])
        prediction = torch.nan_to_num(target, nan=0.) + 2
        counts, errors = bin_sums(prediction, target)
        self.assertEqual(counts.tolist(), [2, 2, 2, 2, 2])
        torch.testing.assert_close(errors, torch.full((5,), 8., dtype=torch.float64))

    def test_empty_bins(self):
        counts, errors = bin_sums(torch.tensor([101.]), torch.tensor([100.]))
        self.assertEqual(counts.tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(errors.tolist(), [0., 1., 0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
