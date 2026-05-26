import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from logging import getLogger
from shutil import rmtree

from src.log import setup_logging


class TestLogger(unittest.TestCase):
    @patch("sys.stderr", new_callable=StringIO)
    @patch("sys.stdout", new_callable=StringIO)
    def test_stdout_and_stderr_routing(self, mock_stdout, mock_stderr):
        log_file = setup_logging("DEBUG", "logs_temps")
        log = getLogger(__name__)

        log.debug("Debug message")
        log.info("Info message")
        log.warning("Warning message")
        log.error("Error message")
        log.critical("Critical message")

        stdout_output = mock_stdout.getvalue()
        stderr_output = mock_stderr.getvalue()

        self._assert_contains(stdout_output, "Debug message", "Info message")
        self._assert_not_contains(
            stdout_output,
            "Warning message",
            "Error message",
            "Critical message",
        )

        self._assert_contains(
            stderr_output,
            "Warning message",
            "Error message",
            "Critical message",
        )
        self._assert_not_contains(
            stderr_output,
            "Debug message",
            "Info message",
        )

        self._assert_contains(stdout_output, __name__, "DEBUG", "INFO")
        self._assert_contains(
            stderr_output,
            __name__,
            "WARNING",
            "ERROR",
            "CRITICAL",
        )

        log_path = Path(log_file)
        self.assertTrue(log_path.exists())
        file_contents = log_path.read_text(encoding="utf-8")
        self._assert_contains(
            file_contents,
            "Debug message",
            "Info message",
            "Warning message",
            "Error message",
            "Critical message",
        )

    @classmethod
    def tearDownClass(cls):
        rmtree("./logs_temps/", ignore_errors=True)

    def _assert_contains(self, text: str, *needles: str) -> None:
        for needle in needles:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def _assert_not_contains(self, text: str, *needles: str) -> None:
        for needle in needles:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, text)


if __name__ == "__main__":
    unittest.main()
