import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from unittest.mock import patch

import train_hybiomass
from train_hybiomass import PreparedDataset, UNet, enmap_stats, evaluate, patch_rmse, save_checkpoint, split_paths, target_stats


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

    def test_evaluation_can_clip_negative_predictions(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor([[[[-1.0]]]]), torch.tensor([[[[1.0]]]]))
        loader = torch.utils.data.DataLoader(dataset)
        result = evaluate(torch.nn.Identity(), loader, torch.device("cpu"), 0, 1, clip_negative=True)
        self.assertEqual(result["rmse_mg_ha"], 1.0)

    def test_atomic_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, {"epoch": 1})
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)
            save_checkpoint(path, {"epoch": 2})
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 2)

    def test_resume_restores_epoch_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            found = {}
            for i in range(100):
                tile = f"{i:05d}"
                found.setdefault(train_hybiomass.split_for_tile(tile), tile)
            for tile in found.values():
                path = root / "data" / "africa" / tile / "patch.npz"
                path.parent.mkdir(parents=True)
                target = np.full((128, 128), np.nan, np.float32)
                target[0, :2] = (1, 3)
                np.savez_compressed(path, embeddings=np.zeros((128, 128, 128), np.float32),
                                    target=target, valid=np.ones((128, 128), bool))
            run = root / "run"
            args = ["train", "--data", str(root / "data"), "--run", str(run), "--device", "cpu",
                    "--workers", "0", "--epochs", "1"]
            with patch.object(train_hybiomass, "UNet", lambda in_channels: torch.nn.Conv2d(in_channels, 1, 1)), patch.object(sys, "argv", args):
                train_hybiomass.main()
            with patch.object(train_hybiomass, "UNet", lambda in_channels: torch.nn.Conv2d(in_channels, 1, 1)), patch.object(sys, "argv", args[:-1] + ["2", "--resume"]):
                train_hybiomass.main()
            checkpoint = torch.load(run / "last.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertTrue(checkpoint["optimizer"]["state"])

    def test_enmap_stats_and_shared_label_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path = root / "prepared/africa/00001/01.npz"
            source_path = root / "enmap/africa/00001/01.npz"
            target_path.parent.mkdir(parents=True)
            source_path.parent.mkdir(parents=True)
            target = np.ones((128, 128), np.float32)
            valid = np.ones((128, 128), bool)
            valid[0, 0] = False
            np.savez_compressed(target_path, target=target, valid=valid)
            image = np.empty((202, 128, 128), np.float32)
            image[:, :, ::2], image[:, :, 1::2] = 1, 3
            image[0, 0, 0] = np.nan
            from prepare_enmap import save_input
            save_input(source_path, image)
            mean, std = enmap_stats([target_path], root / "enmap")
            self.assertAlmostEqual(mean[1], 2.)
            self.assertAlmostEqual(std[1], 1.)
            x, y = PreparedDataset([target_path], 0, 1, root / "enmap", mean, std)[0]
            self.assertEqual(tuple(x.shape), (202, 128, 128))
            self.assertTrue(torch.isfinite(x).all())
            self.assertTrue(torch.isnan(y[0, 0, 0]))

    def test_model_output_shape(self):
        model = UNet().eval()
        with torch.no_grad():
            output = model(torch.zeros((1, 128, 16, 16)))
        self.assertEqual(tuple(output.shape), (1, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
