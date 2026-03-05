import re
import json
from typing import Any, Dict, List, Set

import numpy as np
from pydantic import BaseModel

from src.llm_sdk import Small_LLM_Model
from src.parsing import Parsing

STOP_WORDS: Set[str] = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
    'to', 'of', 'for', 'with', 'what', 'how', 'all', 'by', 'from',
    'this', 'that',
}


class FunctionCallingResult(BaseModel):
    """Validated result of a function calling inference."""

    prompt: str
    name: str
    parameters: Dict[str, Any]


class Model:
    """Wraps a Small_LLM_Model to perform function-calling inference."""

    def __init__(self, parsing: Parsing) -> None:
        """Initialise the model and precompute numeric token ids."""
        self.model = Small_LLM_Model()
        self.parsing = parsing
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as f:
            self.vocab: Dict[str, int] = json.load(f)

        numeric_allowed = "0123456789.-"
        self.numeric_allowed_ids: List[int] = [
            t_id
            for t_str, t_id in self.vocab.items()
            if (
                (cleaned := t_str.replace('Ġ', '').replace('Ċ', ''))
                and all(c in numeric_allowed for c in cleaned)
            )
        ]

        self.system_context_text = (
            "System: You are an expert API router. "
            "Map user requests to the most appropriate function "
            "based on the description.\n\n"
            "Example formats:\n"
            "- User: 'What is 5 plus 3?' -> Function for adding "
            "numbers, parameters: a, b\n"
            "- User: 'Flip the text abc' -> Function for reversing "
            "text, parameter: string\n\n"
            "Available Functions:\n"
        )
        for fn in self.parsing.functions:
            param_str = ', '.join(
                f"{arg}: {fn.parameters[arg].get('type')}"
                for arg in fn.parameters.keys()
            )
            self.system_context_text += f"- {fn.name}({param_str})\n"
            if fn.description:
                self.system_context_text += (
                    f"  Description: {fn.description}\n")

        self.system_ids = self._ensure_flat_list(
            self.model.encode(self.system_context_text))
        self.json_start_ids = self._ensure_flat_list(
            self.model.encode('\nUser Request: '))
        self.json_prompt_key_ids = self._ensure_flat_list(
            self.model.encode('\nJSON response: {"prompt": "'))

        self.min_function = np.log(0.45)

        self.min_function = np.log(0.45)

    def _ensure_flat_list(self, data: Any) -> List[int]:
        """Recursively flatten any nested list or numpy array to List[int]."""
        if hasattr(data, "tolist"):
            data = data.tolist()
        if not isinstance(data, list):
            return [int(data)]
        flat: List[int] = []
        for item in data:
            if isinstance(item, list):
                flat.extend(self._ensure_flat_list(item))
            else:
                flat.append(int(item))
        return flat

    def resolve_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """Resolve a natural-language prompt to a function call dict."""
        ids = list(self.system_ids)
        ids.extend(self.json_start_ids)
        ids.extend(self._ensure_flat_list(self.model.encode(user_prompt)))
        ids.extend(self.json_prompt_key_ids)
        clean_user_prompt = (
            user_prompt.replace("\\", "\\\\").replace('"', '\\"'))
        ids.extend(self._ensure_flat_list(self.model.encode(
            clean_user_prompt)))
        ids.extend(self._ensure_flat_list(self.model.encode(
            '", "name": "')))

        functions = {v.name: v for v in self.parsing.functions}

        ids.extend(self._ensure_flat_list(
            self.model.encode('", "name": "')
        ))

        selected_fn = self._select_function(
            user_prompt, self.parsing.functions, ids
        )

        ids.extend(self._ensure_flat_list(self.model.encode(selected_fn)))
        ids.extend(self._ensure_flat_list(
            self.model.encode('", "parameters": {')
        ))

        try:
            fn_def = next(
                f for f in self.parsing.functions if f.name == selected_fn
            )
            ids.extend(self._generate_parameters(ids, fn_def))
        except StopIteration:
            pass

        ids.extend(self._ensure_flat_list(self.model.encode('}}')))

        full_text = self.model.decode(ids)
        json_str = full_text.split("JSON response:")[-1].strip()

        try:
            data: Dict[str, Any] = json.loads(json_str)
        except Exception:
            return {
                "prompt": user_prompt,
                "name": selected_fn,
                "parameters": {},
            }

        try:
            for k, v in data.get("parameters", {}).items():
                if (functions[selected_fn].parameters[k]["type"] == "number"
                        and isinstance(v, (int))):
                    data["parameters"][k] = float(v)
                if isinstance(v, str):
                    data["parameters"][k] = v.strip()
                    v = data["parameters"][k] = v.strip()
                    if '[' in v and ']' not in v:
                        data["parameters"][k] = v + ']'
                    if '(' in v and ')' not in v:
                        data["parameters"][k] = v + ')'
                    if '{' in v and '}' not in v:
                        data["parameters"][k] = v + '}'
            return FunctionCallingResult(**data).model_dump()
        except Exception:
            return {
                "prompt": user_prompt,
                "name": selected_fn,
                "parameters": {},
            }

    def _select_function(
        self,
        prompt: str,
        functions: List[Any],
        current_ids: List[int],
    ) -> str:
        """Select the most relevant function.

        Uses description word overlap, falling back to logits.
        """
        prompt_lower = prompt.lower()
        prompt_words = set(re.findall(r'\b\w+\b', prompt_lower)) - STOP_WORDS

        scores: Dict[str, int] = {}
        for fn in functions:
            score = 0
            if fn.description:
                desc_lower = fn.description.lower()
                desc_words = (
                    set(re.findall(r'\b\w+\b', desc_lower))
                    - STOP_WORDS
                )
                score += len(prompt_words & desc_words) * 50
                for p_word in prompt_words:
                    for d_word in desc_words:
                        if len(p_word) > 3 and len(d_word) > 3:
                            if p_word in d_word or d_word in p_word:
                                score += 30
            scores[fn.name] = score

        max_score = max(scores.values())
        if max_score > 0:
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) == 1 or sorted_scores[1] < sorted_scores[0]:
                return max(scores, key=lambda k: scores[k])

        return self._decode_limited_choice(
            current_ids, [f.name for f in functions]
        )

    def _decode_limited_choice(
        self, current_ids: List[int], choices: List[str], prompt: str = ""
    ) -> str:
        """Return the choice with the highest average log-prob."""
        scores: List[float] = []
        for choice in choices:
            temp_ids = list(current_ids)
            target_tokens = self._ensure_flat_list(self.model.encode(choice))
            log_prob_sum = 0.0
            for t_id in target_tokens:
                logits = self.model.get_logits_from_input_ids(temp_ids)
                logits_arr = np.array(logits)
                step_logits = (
                    logits_arr[-1] if len(logits_arr.shape) > 1 else logits_arr
                )
                shift = step_logits - np.max(step_logits)
                log_probs = shift - np.log(np.sum(np.exp(shift)) + 1e-10)
                log_prob_sum += float(log_probs[t_id])
                temp_ids.append(t_id)
            scores.append(log_prob_sum / max(len(target_tokens), 1))

        if max(scores) < self.min_function:
            raise ValueError("No function found")

        return choices[int(np.argmax(scores))]

    def _generate_parameters(
        self, current_ids: List[int], fn_def: Any
    ) -> List[int]:
        """Generate token ids for all fn_def parameters given the context."""
        param_ids: List[int] = []
        for i, p_name in enumerate(fn_def.parameters.keys()):
            p_type = fn_def.parameters[p_name].get("type")
            param_ids.extend(
                self._ensure_flat_list(self.model.encode(f'"{p_name}": '))
            )
            if p_type == "string":
                param_ids.extend(
                    self._ensure_flat_list(self.model.encode('"'))
                )
                param_ids.extend(
                    self._generate_until_quote(current_ids + param_ids)
                )
                if not self.model.decode(param_ids).endswith('"'):
                    param_ids.extend(
                        self._ensure_flat_list(self.model.encode('"'))
                    )
            else:
                param_ids.extend(
                    self._generate_numeric(current_ids + param_ids)
                )
            if i < len(fn_def.parameters.keys()) - 1:
                param_ids.extend(
                    self._ensure_flat_list(self.model.encode(", "))
                )
        return param_ids

    def _generate_until_quote(self, current_ids: List[int]) -> List[int]:
        """Generate arguments and stop when current char is quote."""
        generated: List[int] = []

        while True:
            logits = self.model.get_logits_from_input_ids(
                current_ids + generated)
            logits_arr = np.array(logits)
            step_logits = (logits_arr[-1] if
                           len(logits_arr.shape) > 1 else logits_arr)

            next_token = int(np.argmax(step_logits))
            char = self.model.decode([next_token])

            if '",' in char or '"}' in char or '"\n' in char:
                break

            if '"' in char:
                temp_ids = current_ids + generated + [next_token]
                future_logits = self.model.get_logits_from_input_ids(temp_ids)
                future_step_logits = np.array(future_logits)[-1]
                next_char = self.model.decode([
                    int(np.argmax(future_step_logits))])
                if any(stop_char in next_char for stop_char
                       in [',', '}', '\n']) or next_char.strip() in [',', '}']:
                    break
                else:
                    generated.append(next_token)
                    continue

            if '\n' in char:
                break

            generated.append(next_token)

        return generated

    def _generate_numeric(self, current_ids: List[int]) -> List[int]:
        """Greedily generate numeric tokens until a stop character."""
        generated: List[int] = []
        stop_chars = [",", "}", " ", "\n"]
        while True:
            logits = self.model.get_logits_from_input_ids(
                current_ids + generated
            )
            logits_arr = np.array(logits)
            step_logits = (
                logits_arr[-1] if len(logits_arr.shape) > 1 else logits_arr
            )
            mask = np.full_like(step_logits, -np.inf)
            mask[self.numeric_allowed_ids] = 0
            for char in stop_chars:
                stop_id = self.vocab.get(char) or self.vocab.get("Ġ" + char)
                if stop_id:
                    mask[stop_id] = 0
            next_token = int(np.argmax(step_logits + mask))
            char = self.model.decode([next_token])
            if any(c in stop_chars for c in char):
                break
            generated.append(next_token)
        return generated
