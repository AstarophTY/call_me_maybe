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
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def _ensure_flat_list(self, data: Any) -> List[int]:
        if hasattr(data, "tolist"): data = data.tolist()
        if not isinstance(data, list): return [int(data)]
        flat = []
        for item in data:
            if isinstance(item, list): flat.extend(self._ensure_flat_list(item))
            else: flat.append(int(item))
        return flat

    def resolve_prompt(self, user_prompt: str) -> Dict[str, Any]:
        # --- AMÉLIORATION MAJEURE : FEW-SHOT EXAMPLES ---
        # On donne des exemples concrets pour que le modèle de 0.6B comprenne la différence.
        system_context = (
            "System: You are an expert API router. Map requests to functions.\n"
            "Examples:\n"
            "- 'Hello Bob' -> fn_greet(name='Bob')\n"
            "- 'Sum of 1 and 1' -> fn_add_numbers(a=1, b=1)\n"
            "- 'Root of 9' -> fn_get_square_root(a=9)\n\n"
            "Available Functions:\n"
        )
        for fn in self.parsing.functions:
            system_context += f"- {fn.fn_name}({', '.join(fn.args_names)})\n"
        
        full_prompt = f"{system_context}\nUser Request: {user_prompt}\nJSON response:"
        
        ids = self._ensure_flat_list(self.model._encode(full_prompt))
        ids.extend(self._ensure_flat_list(self.model._encode('{"prompt": "')))
        ids.extend(self._ensure_flat_list(self.model._encode(user_prompt)))
        ids.extend(self._ensure_flat_list(self.model._encode('", "name": "')))

        fn_names = [fn.fn_name for fn in self.parsing.functions]
        selected_fn = self._decode_limited_choice(ids, fn_names)
        
        ids.extend(self._ensure_flat_list(self.model._encode(selected_fn)))
        ids.extend(self._ensure_flat_list(self.model._encode('", "parameters": {')))
        
        try:
            fn_def = next(f for f in self.parsing.functions if f.fn_name == selected_fn)
            params_ids = self._generate_parameters(ids, fn_def)
            ids.extend(params_ids)
        except StopIteration: pass
        
        ids.extend(self._ensure_flat_list(self.model._encode('}}')))

        full_text = self.model._decode(ids)
        json_str = full_text.split("JSON response:")[-1].strip()
        
        try:
            data = json.loads(json_str)
            # Nettoyage final pour les types numériques
            for k, v in data["parameters"].items():
                if isinstance(v, float) and abs(v - round(v)) < 1e-7:
                    data["parameters"][k] = int(round(v))
            return FunctionCallingResult(**data).model_dump()
        except:
            return {"prompt": user_prompt, "name": selected_fn, "parameters": {}}

    def _decode_limited_choice(self, current_ids: List[int], choices: List[str]) -> str:
        scores = []
        for choice in choices:
            temp_ids = list(current_ids)
            target_tokens = self._ensure_flat_list(self.model._encode(choice))
            log_prob_sum = 0.0
            
            for t_id in target_tokens:
                logits = self.model.get_logits_from_input_ids(temp_ids)
                logits_arr = np.array(logits)
                step_logits = logits_arr[-1] if len(logits_arr.shape) > 1 else logits_arr
                
                # Normalisation Log-Softmax
                shift = step_logits - np.max(step_logits)
                log_probs = shift - np.log(np.sum(np.exp(shift)) + 1e-10)
                
                log_prob_sum += float(log_probs[t_id])
                temp_ids.append(t_id)
            
            # Normalisation par la longueur pour ne pas pénaliser les noms longs
            scores.append(log_prob_sum / max(len(target_tokens), 1))
        
        return choices[int(np.argmax(scores))]

    def _generate_parameters(self, current_ids: List[int], fn_def: Any) -> List[int]:
        param_ids = []
        for i, p_name in enumerate(fn_def.args_names):
            p_type = fn_def.args_types.get(p_name, "str")
            param_ids.extend(self._ensure_flat_list(self.model._encode(f'"{p_name}": ')))
            
            if p_type in ["str", "string"]:
                param_ids.extend(self._ensure_flat_list(self.model._encode('"')))
                val_ids = self._generate_until_quote(current_ids + param_ids)
                param_ids.extend(val_ids)
                param_ids.extend(self._ensure_flat_list(self.model._encode('"')))
            else:
                # Pour les nombres, on force le masque
                val_ids = self._generate_numeric(current_ids + param_ids)
                param_ids.extend(val_ids)
            
            if i < len(fn_def.args_names) - 1:
                param_ids.extend(self._ensure_flat_list(self.model._encode(", ")))
        return param_ids

    def _generate_until_quote(self, current_ids: List[int]) -> List[int]:
        generated = []
        for _ in range(50):
            logits = self.model.get_logits_from_input_ids(current_ids + generated)
            logits_arr = np.array(logits)
            step_logits = logits_arr[-1] if len(logits_arr.shape) > 1 else logits_arr
            next_token = int(np.argmax(step_logits))
            char = self.model._decode([next_token])
            if '"' in char: break
            generated.append(next_token)
        return generated

    def _generate_numeric(self, current_ids: List[int]) -> List[int]:
        generated = []
        allowed = "0123456789.-"
        for _ in range(15): # Réduit pour éviter les chiffres infinis
            logits = self.model.get_logits_from_input_ids(current_ids + generated)
            logits_arr = np.array(logits)
            step_logits = logits_arr[-1] if len(logits_arr.shape) > 1 else logits_arr
            
            mask = np.full_like(step_logits, -np.inf)
            for t_str, t_id in self.vocab.items():
                c = t_str.replace('Ġ', '').replace('Ċ', '')
                if c and all(char in allowed for char in c):
                    if t_id < len(mask): mask[t_id] = 0
            
            next_token = int(np.argmax(step_logits + mask))
            char = self.model._decode([next_token])
            if not any(c in allowed for c in char): break
            generated.append(next_token)
        return generated