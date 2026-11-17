import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from download_africa import credentials, download


class DownloadTest(unittest.TestCase):
    def test_gzip_and_refuse_overwrite(self):
        response = MagicMock()
        response.iter_content.return_value = iter([b"\x1f\x8bdata"])
        response.headers = {"Content-Length": "6"}
        session = MagicMock()
        session.get.return_value.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "archive.gz"
            download(session, dest)
            self.assertEqual(dest.read_bytes(), b"\x1f\x8bdata")
            with self.assertRaises(FileExistsError):
                download(session, dest)

    def test_displays_progress(self):
        response = MagicMock()
        response.iter_content.return_value = iter([b"\x1f\x8b", b"data"])
        response.headers = {"Content-Length": "6"}
        session = MagicMock()
        session.get.return_value.__enter__.return_value = response
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output):
            with patch("download_africa.monotonic", side_effect=[0, 6, 6]):
                download(session, Path(directory) / "archive.gz")
        self.assertIn("100.0%", output.getvalue())

    def test_credentials_from_env_and_prompt_fallback(self):
        with patch("download_africa.dotenv_values", return_value={"DLR_USERNAME": "user", "DLR_PASSWORD": "secret"}):
            self.assertEqual(credentials(), ("user", "secret"))
        with patch("download_africa.dotenv_values", return_value={}), \
             patch("builtins.input", return_value="user"), \
             patch("download_africa.getpass", return_value="secret"):
            self.assertEqual(credentials(), ("user", "secret"))

    def test_reject_login_page(self):
        response = MagicMock()
        response.iter_content.return_value = iter([b"<html>login</html>"])
        session = MagicMock()
        session.get.return_value.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                download(session, Path(directory) / "archive.gz")


if __name__ == "__main__":
    unittest.main()
