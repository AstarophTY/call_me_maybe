from src.llm_sdk import Small_LLM_Model
import numpy as np
from src.parsing import Parsing
import json


class Model:
    def __init__(self, parsing: Parsing):
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.parsing: Parsing = parsing

        vocab_path = self.model.get_path_to_vocabulary_json()
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.whitelist = []
        for function in parsing.functions:
            token_ids = self.model._encode(function.fn_name)[0].tolist()
            self.whitelist.append(token_ids)

        for prompt in self.parsing.prompts:
            function = self.resolve_prompt(prompt)
            print(function, prompt)

    def change_logits(self) -> None:
        pass

    def optimize_prompt(
        self,
        logits: dict[str, float],
        char: str
    ) -> dict[str, float]:
        token_id = self.vocab.get(char)
        if token_id:
            logits[token_id] = np.inf
        return logits

    def resolve_prompt(self, prompt: str) -> str:
        formatted_prompt = (
            f"Task: Select the best function name for the user request.\n"
            f"Example: 'Add 2 and 5' -> fn_add_numbers\n"
            f"Request: '{prompt}' ->"
        )

        choices = [fn.fn_name for fn in self.parsing.functions]
        best_function = choices[0]
        max_score = -float('inf')

        for fn_name in choices:
            input_ids = self.model._tokenizer.encode(
                formatted_prompt,
                add_special_tokens=False
            )

            target_ids = self.model._tokenizer.encode(
                " " + fn_name,
                add_special_tokens=False
            )

            total_logit = 0
            current_ids = list(input_ids)

            for t_id in target_ids:
                logits = self.model.get_logits_from_input_ids(current_ids)
                total_logit += logits[t_id]
                current_ids.append(t_id)

            avg_score = total_logit / len(target_ids)

            if avg_score > max_score:
                max_score = avg_score
                best_function = fn_name

        return best_function
