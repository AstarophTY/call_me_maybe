import json
import numpy as np
from llm_sdk import Small_LLM_Model


class FunctionCaller:
    def __init__(self, model: Small_LLM_Model):
        self.model = model
        vocab_path = self.model.get_path_to_vocabulary_json()
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.token_to_id = self.vocab

    def get_constrained_logit(self, current_token_ids: list[int], allowed_tokens: list[str]) -> int:
        logits = self.model.get_logits_from_input_ids(current_token_ids)
        allowed_ids = [self.token_to_id[t] for t in allowed_tokens if t in self.token_to_id]
        mask = np.full(len(logits), -np.inf)
        for idx in allowed_ids:
            mask[idx] = logits[idx]
        return int(np.argmax(mask))

    def run(self, user_query: str, functions_list: list[dict]):
        prompt = f"System: Use JSON. Functions: {json.dumps(functions_list)}\nUser: {user_query}\nOutput:"
        input_ids = self.model.encode(prompt)

        # Ici commence ta boucle de génération (Machine à états)
        # 1. Forcer '{'
        # 2. Forcer '"prompt"'
        # ... etc.

        # Simulation d'une sortie pour l'exemple
        # Une fois le JSON généré en string :
        # return FunctionCallResponse.model_validate_json(generated_json_str)