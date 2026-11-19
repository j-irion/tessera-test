import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train_hybiomass import UNet, patch_rmse, split_paths, target_stats


class TrainingTest(unittest.TestCase):
    def test_split_keeps_same_tile_together_across_regions(self):
        paths = [Path("a/00016/1.npz"), Path("b/00016/2.npz")]
        split = split_paths(paths, 42)
        self.assertEqual(sorted(len(p) for p in split.values()), [0, 0, 2])

    def test_masked_loss_ignores_missing_labels(self):
        pred = torch.tensor([[[[1.0, 999.0], [3.0, 2.0]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, float("nan")], [1.0, float("nan")]]]])
        loss = patch_rmse(pred, target)
        self.assertAlmostEqual(loss.item(), 2 ** 0.5, places=5)
        loss.backward()
        self.assertEqual(pred.grad[0, 0, 0, 1].item(), 0)

    def test_stats_exclude_missing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.npz"
            np.savez(path, target=np.array([1., np.nan, 3.]))
            self.assertEqual(target_stats([path]), (2.0, 1.0))

    def test_model_output_shape(self):
        model = UNet().eval()
        with torch.no_grad():
            output = model(torch.zeros((1, 128, 16, 16)))
        self.assertEqual(tuple(output.shape), (1, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
