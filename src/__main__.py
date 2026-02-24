from src.model import Model
from src.parsing import Parsing


def main() -> None:
    parsing = Parsing()
    model = Model(parsing)


if __name__ == "__main__":
    main()
