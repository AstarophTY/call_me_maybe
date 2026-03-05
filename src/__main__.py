import json
from src.model import Model
from src.parsing import Parsing
from tqdm import tqdm
from sys import stderr


def main() -> None:
    """Main function
    """
    try:
        parsing = Parsing()
    except Exception as e:
        print(f"Error on parsing: {e}", file=stderr)
        return

    engine = Model(parsing)

    results = []
    with tqdm(desc="Process prompt", total=len(parsing.prompts)) as progress:
        for prompt_text in parsing.prompts:
            try:
                result = engine.resolve_prompt(prompt_text)
                results.append(result)
            except Exception as e:
                print(f"Error on prompt: {e}", file=stderr)
            progress.update(1)

    time = progress.format_dict['elapsed']

    try:
        with open(parsing.output, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error: \n{e}", file=stderr)
    else:
        print(f"Done! Results saved to {parsing.output} in {round(time, 1)}s")


if __name__ == "__main__":
    main()
