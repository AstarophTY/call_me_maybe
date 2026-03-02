import json
from src.model import Model
from src.parsing import Parsing
from sys import stderr


def main() -> None:
    """
        Main function
    """
    try:
        parsing = Parsing()
    except Exception:
        return

    engine = Model(parsing)

    results = []

    for prompt_text in parsing.prompts:
        print(f"Processing: {prompt_text[:50]}...")
        try:
            result = engine.resolve_prompt(prompt_text)
            results.append(result)
        except Exception as e:
            print(f"Error on prompt: {e}", file=stderr)

    try:
        with open(parsing.output, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error: \n{e}", file=stderr)
    else:
        print(f"Done! Results saved to {parsing.output}")


if __name__ == "__main__":
    main()
