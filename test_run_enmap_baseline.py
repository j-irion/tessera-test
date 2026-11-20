import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class EnmapRunnerTest(unittest.TestCase):
    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy("run_enmap_baseline.sh", root)
            (root / "data/runs/enmap_unet_01").mkdir(parents=True)
            result = subprocess.run(["bash", "run_enmap_baseline.sh"], cwd=root,
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)


if __name__ == "__main__":
    unittest.main()
