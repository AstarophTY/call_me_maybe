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

        self.optimize_model()
        self.resolve_prompts()

    def change_logits(self) -> None:
        pass

    def optimize_model(self) -> None:
        print(self.vocab.get("{"))

    def resolve_prompts(self) -> None:
        for prompt in self.parsing.prompts:
            input_ids = self.model._encode(prompt).tolist()[0]
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token_id = int(np.argmax(logits))
            predicted_word = self.model._decode([next_token_id])
            print(f"Prompt: {prompt}")
            print(f"Token suivant prédit: '{predicted_word}' (ID: {next_token_id})")
