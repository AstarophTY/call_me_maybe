from pydantic import ValidationError
from argparse import ArgumentParser
from src.validator import ParsingValidation


class Parsing:
    def __init__(self):
        parser = ArgumentParser(description="Call Me Maybe")

        parser.add_argument("--functions_definition", type=str, help="")
        parser.add_argument("--input", type=str)
        parser.add_argument("--output", type=str)

        args = parser.parse_args()
        clean_args = {k: v for k, v in vars(args).items() if v is not None}

        try:
            values = ParsingValidation(**clean_args)
        except ValidationError as e:
            print(f"Error: \n{e}")

        self.prompts = [str(prompt) for prompt in values.prompts]
        self.functions = list(values.functions)
        self.output = str(values.output)
