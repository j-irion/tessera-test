import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import download_remaining


class RemainingDownloadTest(unittest.TestCase):
    def test_skips_existing_and_downloads_remaining(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                Path("data").mkdir()
                Path("data/hybiomass_europe.tar.gz").write_bytes(b"existing")
                with patch("builtins.input", return_value="user"), \
                     patch.object(download_remaining, "getpass", return_value="password"), \
                     patch.object(download_remaining, "sign_in"), \
                     patch.object(download_remaining, "download") as download:
                    download_remaining.main()
                self.assertEqual(download.call_count, 5)
                self.assertFalse(any("europe" in str(call) for call in download.call_args_list))
                self.assertEqual(Path("data/hybiomass_europe.tar.gz").read_bytes(), b"existing")
            finally:
                os.chdir(previous)

    def test_partial_file_stops_before_login(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                Path("data").mkdir()
                Path("data/hybiomass_europe.tar.gz.part").touch()
                with patch("builtins.input") as prompt:
                    with self.assertRaises(FileExistsError):
                        download_remaining.main()
                    prompt.assert_not_called()
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
