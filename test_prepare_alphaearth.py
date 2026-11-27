import unittest

import numpy as np

from prepare_alphaearth import normalize_vectors


class AlphaEarthTest(unittest.TestCase):
    def test_unit_vector_normalization_and_invalid_mask(self):
        values = np.array([[[3., 1.], [1., 0.]], [[4., 0.], [1., 0.]]], dtype=np.float32)
        result, valid = normalize_vectors(values, np.array([[True, True], [True, False]]))
        np.testing.assert_allclose(result[:, 0, 0], [0.6, 0.8])
        np.testing.assert_allclose(result[:, 1, 0], [1 / np.sqrt(2), 1 / np.sqrt(2)])
        self.assertTrue(valid.tolist() == [[True, True], [True, False]])
        np.testing.assert_array_equal(result[:, 1, 1], [0, 0])


if __name__ == "__main__":
    unittest.main()
