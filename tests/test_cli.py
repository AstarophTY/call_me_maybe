import os
import unittest
from unittest import mock
from typer.testing import CliRunner

from src.cli.main import app


runner = CliRunner()


class TestCLI(unittest.TestCase):
    @mock.patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_cli_default_args(self):
        result = runner.invoke(app, [])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("functions_definition.json", result.output)
        self.assertIn("function_calling_tests.json", result.output)
        self.assertIn("function_calling_results.json", result.output)

    @mock.patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_cli_custom_args(self):
        result = runner.invoke(
            app,
            [
                "-f",
                "custom/functions.json",
                "-i",
                "custom/input.json",
                "-o",
                "custom/output.json",
            ],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("custom/functions.json", result.output)
        self.assertIn("custom/input.json", result.output)
        self.assertIn("custom/output.json", result.output)

    @mock.patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_log_level_debug(self):
        result = runner.invoke(app, [])
        self.assertEqual(result.exit_code, 0)

    @mock.patch.dict(os.environ, {"LOG_LEVEL": "INVALID"})
    def test_invalid_log_level(self):
        result = runner.invoke(app, [])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("LOG_LEVEL is not valid", result.output)
        self.assertIn("Using INFO instead", result.output)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_missing_log_level(self):
        result = runner.invoke(app, [])
        self.assertEqual(result.exit_code, 0)

    def test_help_command(self):
        result = runner.invoke(app, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("functions_definition", result.output)
        self.assertIn("input", result.output)
        self.assertIn("output", result.output)

    def test_invalid_option(self):
        result = runner.invoke(app, ["--unknown"])

        self.assertNotEqual(result.exit_code, 0)

    @mock.patch("src.cli.main.setup_logging")
    def test_setup_logging_called(self, mock_setup):
        runner.invoke(app, [])

        mock_setup.assert_called_once()

    @mock.patch("src.cli.main.getenv", return_value="INFO")
    def test_getenv_called(self, mock_getenv):
        runner.invoke(app, [])

        mock_getenv.assert_called_with("LOG_LEVEL", "INFO")


if __name__ == "__main__":
    unittest.main()
