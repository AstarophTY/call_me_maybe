import typer
from logging import getLogger
from ..log import setup_logging
from dotenv import load_dotenv
from os import getenv

app = typer.Typer()


@app.command()
def main(
    functions_definition: str = typer.Option(
        "data/input/functions_definition.json", "-f", "--functions_definition"
    ),
    input: str = typer.Option(
        "data/input/function_calling_tests.json", "-i", "--input"
    ),
    output: str = typer.Option(
        "data/output/function_calling_results.json", "-o", "--output"
    ),
) -> None:
    """
    Main entry point for the function calling CLI application.
    Loads environment configuration, initializes logging, and prepares the \
        application
    for processing function definitions and test cases.

    Args:
        functions_definition (str): Path to the JSON file containing function
            definitions. Defaults to "data/input/functions_definition.json".
        input (str): Path to the JSON file containing function calling test
            cases. Defaults to "data/input/function_calling_tests.json".
        output (str): Path to the JSON file where results will be written.
            Defaults to "data/output/function_calling_results.json".

    Returns:
        None

    Raises:
        Logs a warning if LOG_LEVEL environment variable is invalid.
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
        f"\t- functions_definition: {functions_definition}\n"
        f"\t- input: {input}\n"
        f"\t- output: {output}"
    )
