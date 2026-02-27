import json
import numpy as np
from typing import List, Dict, Any
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
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.numeric_allowed_ids = []
        numeric_allowed = "0123456789.-"
        for t_str, t_id in self.vocab.items():
            cleant_t = t_str.replace('Ġ', '').replace('Ċ', '')
            if cleant_t and all(c in numeric_allowed for c in cleant_t):
                self.numeric_allowed_ids.append(t_id)

    def _ensure_flat_list(self, data: Any) -> List[int]:
        if hasattr(data, "tolist"):
            data = data.tolist()
        if not isinstance(data, list):
            return [int(data)]
        flat = []
        for item in data:
            if isinstance(item, list):
                flat.extend(self._ensure_flat_list(item))
            else:
                flat.append(int(item))
        return flat

    def resolve_prompt(self, user_prompt: str) -> Any:
        system_context = (
            "System: You are an expert API router. \
Map requests to functions.\n"
            "Examples:\n"
            "- 'Hello Bob' -> fn_greet(name='Bob')\n"
            "- 'Sum of 1 and 1' -> fn_add_numbers(a=1, b=1)\n"
            "- 'Root of 9' -> fn_get_square_root(a=9)\n\n"
            "Available Functions:\n"
        )
        for fn in self.parsing.functions:
            system_context += f"- {fn.fn_name}({', '.join(fn.args_names)})\n"

        full_prompt = f"{system_context}\n\
User Request: {user_prompt}\nJSON response:"

        ids = self._ensure_flat_list(self.model._encode(full_prompt))
        ids.extend(self._ensure_flat_list(self.model._encode('{"prompt": "')))
        ids.extend(self._ensure_flat_list(self.model._encode(user_prompt)))
        ids.extend(self._ensure_flat_list(self.model._encode('", "name": "')))

        fn_names = [fn.fn_name for fn in self.parsing.functions]
        selected_fn = self._decode_limited_choice(ids, fn_names)

        ids.extend(self._ensure_flat_list(self.model._encode(selected_fn)))
        ids.extend(self._ensure_flat_list(self.model._encode(
            '", "parameters": {')))

        try:
            fn_def = next(f for f in self.parsing.functions
                          if f.fn_name == selected_fn)
            params_ids = self._generate_parameters(ids, fn_def)
            ids.extend(params_ids)
        except StopIteration:
            pass

        ids.extend(self._ensure_flat_list(self.model._encode('}}')))

        full_text = self.model._decode(ids)
        json_str = full_text.split("JSON response:")[-1].strip()

        try:
            data = json.loads(json_str)
            for k, v in data["parameters"].items():
                if isinstance(v, float) and abs(v - round(v)) < 1e-7:
                    data["parameters"][k] = int(round(v))
            return FunctionCallingResult(**data).model_dump()
        except Exception:
            return {
                "prompt": user_prompt,
                "name": selected_fn,
                "parameters": {}
            }

    def _decode_limited_choice(self, current_ids: List[int],
                               choices: List[str]) -> str:
        scores = []
        for choice in choices:
            temp_ids = list(current_ids)
            target_tokens = self._ensure_flat_list(self.model._encode(choice))
            log_prob_sum = 0.0

            for t_id in target_tokens:
                logits = self.model.get_logits_from_input_ids(temp_ids)
                logits_arr = np.array(logits)
                step_logits = (logits_arr[-1] if len(logits_arr.shape) > 1
                               else logits_arr)

                shift = step_logits - np.max(step_logits)
                log_probs = shift - np.log(np.sum(np.exp(shift)) + 1e-10)

                log_prob_sum += float(log_probs[t_id])
                temp_ids.append(t_id)

            scores.append(log_prob_sum / max(len(target_tokens), 1))

        return choices[int(np.argmax(scores))]

    def _generate_parameters(self, current_ids: List[int],
                             fn_def: Any) -> List[int]:
        param_ids = []
        for i, p_name in enumerate(fn_def.args_names):
            p_type = fn_def.args_types.get(p_name, "str")
            param_ids.extend(
                self._ensure_flat_list(self.model._encode(f'"{p_name}": ')))

            if p_type in ["str"]:
                param_ids.extend(
                    self._ensure_flat_list(self.model._encode('"')))
                val_ids = self._generate_until_quote(current_ids + param_ids)
                param_ids.extend(val_ids)
                param_ids.extend(
                    self._ensure_flat_list(self.model._encode('"')))
            else:
                val_ids = self._generate_numeric(current_ids + param_ids)
                param_ids.extend(val_ids)

            if i < len(fn_def.args_names) - 1:
                param_ids.extend(
                    self._ensure_flat_list(self.model._encode(", ")))
        return param_ids

    def _generate_until_quote(self, current_ids: List[int]) -> List[int]:
        generated: List[int] = []

        for _ in range(50):
            logits = self.model.get_logits_from_input_ids(
                current_ids + generated)
            logits_arr = np.array(logits)
            step_logits = (logits_arr[-1] if len(logits_arr.shape) > 1
                           else logits_arr)
            quote_id = self.vocab.get('"', -1)
            if quote_id != -1:
                step_logits[quote_id] = -np.inf
            next_token = int(np.argmax(step_logits))
            char = self.model._decode([next_token])
            if '"' in char or '\n' in char:
                break
            generated.append(next_token)
        return generated

    def _generate_numeric(self, current_ids: List[int]) -> List[int]:
        generated: List[int] = []
        stop_chars = [",", "}", " ", "\n"]

        for _ in range(15):
            logits = self.model.get_logits_from_input_ids(
                current_ids + generated)
            logits_arr = np.array(logits)
            step_logits = (logits_arr[-1] if len(logits_arr.shape) > 1
                           else logits_arr)

            mask = np.full_like(step_logits, -np.inf)
            mask[self.numeric_allowed_ids] = 0

            for char in stop_chars:
                stop_id = self.vocab.get(char) or self.vocab.get("Ġ" + char)
                if stop_id:
                    mask[stop_id] = 0

            next_token = int(np.argmax(step_logits + mask))
            char = self.model._decode([next_token])
            if not any(c in stop_chars for c in char):
                break
            generated.append(next_token)
        return generated
