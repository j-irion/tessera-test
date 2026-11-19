import tempfile
import unittest
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

from prepare_hybiomass import aggregate_embeddings, save_patch


class AlignmentTest(unittest.TestCase):
    def test_averages_aligned_10m_pixels_and_masks_missing(self):
        values = np.arange(36, dtype=np.float32).reshape(6, 6)
        vectors = np.repeat(values[..., None], 128, axis=2)
        vectors[0, 0] = np.nan
        result, valid = aggregate_embeddings(vectors, from_origin(0, 60, 10, 10), "EPSG:32632",
                                             (2, 2), from_origin(0, 60, 30, 30), "EPSG:32632")
        self.assertEqual(valid.tolist(), [[False, True], [True, True]])
        self.assertEqual(result[0, 0, 0], 0)
        self.assertAlmostEqual(result[0, 0, 1], values[:3, 3:].mean())
        self.assertAlmostEqual(result[127, 1, 0], values[3:, :3].mean())

    def test_target_grid_offset_is_respected(self):
        values = np.broadcast_to(np.arange(9, dtype=np.float32)[None, :, None], (6, 9, 128)).copy()
        result, valid = aggregate_embeddings(values, from_origin(0, 60, 10, 10), "EPSG:32632",
                                             (1, 1), from_origin(20, 60, 30, 30), "EPSG:32632")
        self.assertTrue(valid[0, 0])
        self.assertAlmostEqual(result[0, 0, 0], 3.0)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.npz"
            args = (np.zeros((128, 2, 2)), np.ones((2, 2)), np.ones((2, 2), bool),
                    2023, "EPSG:32632", from_origin(0, 60, 30, 30))
            save_patch(path, *args)
            with self.assertRaises(FileExistsError):
                save_patch(path, *args)


if __name__ == "__main__":
    unittest.main()
