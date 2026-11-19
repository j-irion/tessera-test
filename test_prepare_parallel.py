import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class ParallelRunnerTest(unittest.TestCase):
    def test_runs_each_region_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy("prepare_parallel_hybiomass.sh", root)
            python = root / ".venv-v11/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text('#!/usr/bin/env bash\necho "$3" >> regions.log\nif [[ "$3" == "$FAIL_REGION" ]]; then exit 1; fi\n')
            python.chmod(0o755)
            environment = {**os.environ, "FAIL_REGION": ""}
            result = subprocess.run(["bash", "prepare_parallel_hybiomass.sh"], cwd=root,
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set((root / "regions.log").read_text().splitlines()),
                             {"africa", "australasia", "europe", "north_america", "north_asia", "south_america", "south_asia"})
            environment["FAIL_REGION"] = "europe"
            failed = subprocess.run(["bash", "prepare_parallel_hybiomass.sh"], cwd=root,
                                    env=environment, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertNotIn("All HyBiomass regions prepared.", failed.stdout)


if __name__ == "__main__":
    unittest.main()
