import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from download_africa import credentials, download


class DownloadTest(unittest.TestCase):
    def test_credentials_from_env_and_prompt_fallback(self):
        with patch("download_africa.dotenv_values", return_value={"DLR_USERNAME": "user", "DLR_PASSWORD": "secret"}):
            self.assertEqual(credentials(), ("user", "secret"))
        with patch("download_africa.dotenv_values", return_value={}), \
             patch("builtins.input", return_value="user"), \
             patch("download_africa.getpass", return_value="secret"):
            self.assertEqual(credentials(), ("user", "secret"))

    def test_download_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "archive.gz"
            partial = Path(str(dest) + ".part")

            def run(args, check):
                self.assertIn("--split=4", args)
                self.assertIn("--max-concurrent-downloads=1", args)
                partial.write_bytes(b"\x1f\x8bdata")

            with patch("download_africa.subprocess.run", side_effect=run):
                download(MagicMock(cookies=[]), dest)
            self.assertEqual(dest.read_bytes(), b"\x1f\x8bdata")
            self.assertFalse(partial.exists())
            with self.assertRaises(FileExistsError):
                download(MagicMock(cookies=[]), dest)

    def test_existing_partial_not_from_aria2_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "archive.gz"
            partial = Path(str(dest) + ".part")
            partial.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                download(MagicMock(cookies=[]), dest)
            self.assertEqual(partial.read_bytes(), b"old")


if __name__ == "__main__":
    unittest.main()
