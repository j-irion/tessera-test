import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from download_africa import download


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
