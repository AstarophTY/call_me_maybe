import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.parsing import ConfigLoader


class TestParsing(unittest.TestCase):
    def test_loads_both_json_files_with_pydantic(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            loader = ConfigLoader(
                input_file="data/input/function_calling_tests.json",
                output_file=str(output_dir),
                functions_definition_file=(
                    "data/input/functions_definition.json"
                ),
            )

            input_data = loader.data["input_file"]
            functions_definition_data = loader.data[
                "functions_definition_file"
            ]

            self.assertEqual(len(input_data.root), 11)
            self.assertEqual(
                input_data.root[0].prompt,
                "What is the sum of 2 and 3?",
            )

            self.assertEqual(len(functions_definition_data.root), 5)
            self.assertEqual(
                functions_definition_data.root[0].name,
                "fn_add_numbers",
            )
            self.assertEqual(
                functions_definition_data.root[0].parameters["a"].type,
                "number",
            )


if __name__ == "__main__":
    unittest.main()
