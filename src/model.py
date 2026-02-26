import json
from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel
from src.llm_sdk import Small_LLM_Model
from src.parsing import Parsing


class FunctionCallingResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


class Model:
    def __init__(self, parsing: Parsing):
        self.model = Small_LLM_Model()
        self.parsing = parsing
        vocab_path = self.model.get_path_to_vocabulary_json()
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab: Dict[str, int] = json.load(f)

    def _safe_encode(self, text: str) -> List[int]:
        res: Any = self.model._encode(text)
        if hasattr(res, "tolist"):
            res = res.tolist()

        def flatten(items: Any) -> List[int]:
            flat_list = []
            if isinstance(items, list):
                for item in items:
                    flat_list.extend(flatten(item))
            else:
                flat_list.append(int(items))
            return flat_list

        return flatten(res)

    def encode_string_strictly(self, ids: List[int], text: str) -> None:
        for char in text:
            tokens = self._safe_encode(char)
            if tokens:
                ids.append(tokens[0])

    def resolve_prompt(self, user_prompt: str) -> Dict[str, Any]:
        ids = self._safe_encode(f"Request: {user_prompt}\nJSON:")
        self.encode_string_strictly(ids, '{"prompt": "')
        ids.extend(self._safe_encode(user_prompt))
        self.encode_string_strictly(ids, '", "name": "')

        fn_names = [fn.fn_name for fn in self.parsing.functions]
        selected_fn = self._decode_limited_choice(ids, fn_names)

        self.encode_string_strictly(ids, selected_fn)
        self.encode_string_strictly(ids, '", "parameters": {}')
        self.encode_string_strictly(ids, "}")

        full_text = str(self.model._decode(ids))
        json_str = full_text.split("JSON:")[-1]
        data = json.loads(json_str)
        return FunctionCallingResult(**data).model_dump()

    def _decode_limited_choice(self, current_ids: List[int],
                               choices: List[str]) -> str:
        scores: List[float] = []
        for choice in choices:
            temp_ids = list(current_ids)
            target_tokens = self._safe_encode(choice)
            total_logit = 0.0
            for t_id in target_tokens:
                logits = self.model.get_logits_from_input_ids(temp_ids)
                val = logits[t_id]
                if hasattr(val, "item"):
                    total_logit += float(val.item())
                else:
                    total_logit += float(val)
                temp_ids.append(t_id)
            scores.append(total_logit / len(target_tokens))
        return choices[int(np.argmax(scores))]
