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
            token_ids = self.model._encode(function.fn_name)
            self.whitelist.append(token_ids)

        self.resolve_prompt(parsing.prompts[0])

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

    def resolve_prompt(self, prompt: str) -> None:
        input_ids = list(self.model._encode(prompt)[0])

        logits = self.model.get_logits_from_input_ids(input_ids)
        restricted_logits = np.full_like(logits, -np.inf)
        function = ""
        for wl_word in self.whitelist:
            for ids in wl_word:
                for id in ids:
                    idx = int(id)
                    restricted_logits[idx] = logits[idx]
                    ids_response = np.argmax(restricted_logits)
                    input_ids.append(ids_response)
                    word = self.model._decode([int(ids_response)])
                    # restricted_logits[idx] -= 5.0
                    function += word
        print(function)
