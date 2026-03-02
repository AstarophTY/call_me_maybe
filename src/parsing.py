from pydantic import ValidationError
from argparse import ArgumentParser
from src.validator import ParsingValidation
from sys import stderr


class Parsing:
    def __init__(self) -> None:
        """Build parsing object."""
        parser = ArgumentParser(description="Call Me Maybe")

        parser.add_argument("--functions_definition", type=str, help="")
        parser.add_argument("--input", type=str)
        parser.add_argument("--output", type=str)

        args = parser.parse_args()
        clean_args = {k: v for k, v in vars(args).items() if v is not None}

        try:
            values = ParsingValidation(**clean_args)
        except ValidationError as e:
            print("\nValidation Error:", file=stderr)
            for error in e.errors():
                message = error['msg']
                if message.startswith("Value error, "):
                    message.replace("Value error, ", "", 1)
                print(f"Error: {message}", file=stderr)

        self.prompts = [prompt.prompt for prompt in values.prompts]
        self.functions = list(values.functions)
        self.output = str(values.output)
