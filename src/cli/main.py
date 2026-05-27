import typer
from logging import getLogger
from dotenv import load_dotenv
from os import getenv

from ..log import setup_logging
from ..parsing.config_loader import ConfigLoader

app = typer.Typer()


@app.command()
def main(
    functions_definition_file: str = typer.Option(
        "data/input/functions_definition.json",
        "-f",
        "--functions_definition",
    ),
    input_file: str = typer.Option(
        "data/input/function_calling_tests.json",
        "-i",
        "--input",
    ),
    output_file: str = typer.Option(
        "data/output/function_calling_results.json",
        "-o",
        "--output",
    ),
) -> None:
    """Main entry point for the function calling CLI.

    Loads environment configuration, initializes logging, and
    processes function definitions and test cases from JSON files.

    :param functions_definition_file: Path to function definitions
        JSON.
    :param input_file: Path to function calling tests JSON.
    :param output_file: Path to output results JSON.
    """
    load_dotenv()

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    raw_log_level = getenv("LOG_LEVEL", "INFO").upper()
    invalid_log_level = raw_log_level not in valid_levels
    log_level = raw_log_level if not invalid_log_level else "INFO"

    setup_logging(log_level)
    log = getLogger(__name__)

    if invalid_log_level:
        log.warning("LOG_LEVEL is not valid. Using INFO instead.")

    log.debug(
        "The program launches with these arguments :\n"
        f"\t- functions_definition_file: {functions_definition_file}\n"
        f"\t- input_file: {input_file}\n"
        f"\t- output_file: {output_file}"
    )

    _ = ConfigLoader(
        input_file=input_file,
        output_file=output_file,
        functions_definition_file=functions_definition_file
    )
