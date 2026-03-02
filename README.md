*This project has been created as part of the 42 curriculum by sgil--de*

# Call Me Maybe


## Description

### The project aims to improve correct answer rates, so to do this we'll use constraint decoding. This allows us to constrain the AI ​​to increase the probability of certain words and decrease it for others.

## Instructions

### To install package for the llm and the project
```bash
make install
```
### To run the program with default path
```bash
make
```

#### Or with args


| Flags | Example | Description |
|----------|:-------------:|------:|
| --functions_definition | data/input/functions_definition.json | These is the path of function definitions json |
| --input | data/input/function_calling_tests.json | These is the path of prompts json |
| --output | data/input/function_calls.json | These is the path of output folder (response) |


```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json
```
### Run Project is debug mode
```bash
make debug
```

### Run typing test
```bash
make lint
```

### Remove temps files
```bash
make clean
```

### Run typing test with strict mode
```bash
make lint-strict
```


## Resource

#### [Pytorch documentation](https://docs.pytorch.org/tutorials/index.html)
#### [Pydantic documentation](https://docs.pydantic.dev/latest/)
#### [UV documentation](https://docs.astral.sh/uv/)

### AI was used
#### For questions on constraint decoding

## Algorithm explanation and Design decisions
We force the JSON structure until we reach a point where a choice must be made. For function selection, I use a regex to split the text; if a word from the prompt is found in the description, I increase its score. If there is a clear winner with no tied scores, I return the one with the highest probability. Otherwise, I use a function that returns the most likely choice from a restricted list of words. For arguments, I force double quotes if the type is a string, and I have a function that generates the argument until it hits a double quote, iterating as long as it doesn't produce a termination character. For numbers, it works similarly, except I force the allowed characters to be digits, dots, or the minus sign.

## Performance analysis
For performances, I generate by default the numbers to avoid iterating each time I look for an argument and for which reliability I have avoided as much as possible to base myself on the raw function names in the code so that it is reusable if the JSON is well made.

## Challenges faced
For this project, I developed a robust Function Calling system using a constrained decoding approach. Instead of letting the model generate text freely, I enforce the JSON structure by injecting the syntax tokens myself. To choose the correct function, I use a hybrid method: I slice the text with a regular expression to compare the words in the prompt with the descriptions, and if the match is close, I let the model choose based on its probabilities (logits). For arguments, I ensure data reliability by enforcing quotation marks for strings and using a logit mask for numbers (allowing only digits, periods, and minus signs). This results in 100% valid JSON every time, even with a small model.

## Testing strategy
For testing this project, I changed the json to verify if the model can select good functions and arguments, and I tried to create an error on the json parameter, putting unknown type and not good format.

• Example usage: Provide clear examples of running your program
## Example usage

#### Add functions in json like:
```json
    {
        "name": "fn_get_weather",
        "description": "Get the current weather for a specific city.",
        "parameters": {
          "city": {
            "type": "string"
          },
        },
        "returns": {
          "type": "string"
        }
    }
```

#### And example parsing we take the function in the json
```python
from src.parsing import Parsing
from src.model import Model

parsing = Parsing()
model = Model(parsing)

prompt_1 = "Check the weather in Paris in celsius"
result_1 = model.resolve_prompt(prompt_1)
print(f"Prompt: {prompt_1}")
print(f"Result: {result_1}\n")
```