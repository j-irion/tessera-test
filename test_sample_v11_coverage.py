import unittest

import pandas as pd

from sample_v11_coverage import selected_patches


class SamplingTest(unittest.TestCase):
    def test_sampling_is_repeatable_and_uses_one_row_per_patch(self):
        frame = pd.DataFrame([
            {"patch": str(i), "variant": variant, "year": year, "enmap_year": 2023}
            for i in range(150) for variant in ("beta1", "beta2") for year in (2022, 2023)
        ])
        sample = selected_patches(frame, 42)
        self.assertEqual(len(sample), 100)
        self.assertEqual(sample, selected_patches(frame.sample(frac=1), 42))


if __name__ == "__main__":
    unittest.main()
