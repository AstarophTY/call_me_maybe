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

    def resolve_prompt(self, user_prompt: str) -> Dict[str, Any]:
        # --- AMÉLIORATION DU PROMPT ---
        # On donne un rôle clair à chaque fonction pour éviter la confusion
        context = "System: You are a precise function selector. Choose the specialized function that matches the user's intent.\n"
        context += "Available Tools:\n"
        for fn in self.parsing.functions:
            # On montre la signature pour aider le modèle à voir les arguments attendus
            context += f"- {fn.fn_name}({', '.join(fn.args_names)})\n"
        
        # On structure la fin du prompt pour "préparer" le terrain au JSON
        full_prompt = f"{context}\nUser Request: {user_prompt}\nResponse Strategy: Select the most specific tool. For math, use math functions. For greetings, use greet. For regex/replace, use substitute.\nJSON response:"
        
        ids = self._ensure_flat_list(self.model._encode(full_prompt))
        
        # Début de la structure JSON forcée
        ids.extend(self._ensure_flat_list(self.model._encode('{"prompt": "')))
        ids.extend(self._ensure_flat_list(self.model._encode(user_prompt)))
        ids.extend(self._ensure_flat_list(self.model._encode('", "name": "')))

        # Sélection de la fonction avec le nouveau scoring
        fn_names = [fn.fn_name for fn in self.parsing.functions]
        selected_fn = self._decode_limited_choice(ids, fn_names)
        
        ids.extend(self._ensure_flat_list(self.model._encode(selected_fn)))
        ids.extend(self._ensure_flat_list(self.model._encode('", "parameters": {')))
        
        # Génération des paramètres
        try:
            fn_def = next(f for f in self.parsing.functions if f.fn_name == selected_fn)
            params_ids = self._generate_parameters(ids, fn_def)
            ids.extend(params_ids)
        except StopIteration:
            pass
        
        ids.extend(self._ensure_flat_list(self.model._encode('}}')))

        # Extraction finale
        full_text = self.model._decode(ids)
        json_str = full_text.split("JSON response:")[-1].strip()
        
        try:
            data = json.loads(json_str)
            return FunctionCallingResult(**data).model_dump()
        except:
            return {"prompt": user_prompt, "name": selected_fn, "parameters": {}}

    def _decode_limited_choice(self, current_ids: List[int], choices: List[str]) -> str:
        """
        Calcule la probabilité de chaque nom de fonction en utilisant log-softmax.
        C'est beaucoup plus précis pour un petit modèle (Qwen 0.6B).
        """
        scores = []
        for choice in choices:
            temp_ids = list(current_ids)
            target_tokens = self._ensure_flat_list(self.model._encode(choice))
            log_prob_sum = 0.0
            
            for t_id in target_tokens:
                logits = self.model.get_logits_from_input_ids(temp_ids)
                
                # On récupère le dernier vecteur de logits
                if hasattr(logits, "shape") and len(logits.shape) > 1:
                     step_logits = logits[-1]
                else:
                     step_logits = logits

                # Calcul du Log-Softmax pour normaliser les scores
                # log_softmax(x) = x - log(sum(exp(x)))
                shift_logits = step_logits - np.max(step_logits)
                log_prob = shift_logits[t_id] - np.log(np.sum(np.exp(shift_logits)) + 1e-10)
                
                log_prob_sum += float(log_prob)
                temp_ids.append(t_id)
            
            # On moyenne par la longueur pour ne pas favoriser les noms courts
            scores.append(log_prob_sum / max(len(target_tokens), 1))
        
        return choices[int(np.argmax(scores))]

    def _generate_parameters(self, current_ids: List[int], fn_def: Any) -> List[int]:
        param_ids = []
        arg_names = fn_def.args_names
        
        for i, p_name in enumerate(arg_names):
            p_type = fn_def.args_types.get(p_name, "str")
            field_str = f'"{p_name}": '
            param_ids.extend(self._ensure_flat_list(self.model._encode(field_str)))
            
            if p_type in ["str", "string"]:
                param_ids.extend(self._ensure_flat_list(self.model._encode('"')))
                val_ids = self._generate_until_quote(current_ids + param_ids)
                param_ids.extend(val_ids)
                param_ids.extend(self._ensure_flat_list(self.model._encode('"')))
            else:
                val_ids = self._generate_numeric(current_ids + param_ids)
                param_ids.extend(val_ids)
            
            if i < len(arg_names) - 1:
                param_ids.extend(self._ensure_flat_list(self.model._encode(", ")))
        
        return param_ids

    def _generate_until_quote(self, current_ids: List[int]) -> List[int]:
        generated = []
        for _ in range(50):
            logits = self.model.get_logits_from_input_ids(current_ids + generated)
            if hasattr(logits, "shape") and len(logits.shape) > 1:
                logits = logits[-1]
            next_token = int(np.argmax(logits))
            char = self.model._decode([next_token])
            if '"' in char:
                break
            generated.append(next_token)
        return generated

    def _generate_numeric(self, current_ids: List[int]) -> List[int]:
        generated = []
        allowed = "0123456789.-"
        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(current_ids + generated)
            if hasattr(logits, "shape") and len(logits.shape) > 1:
                logits = logits[-1]

            mask = np.full_like(logits, -np.inf)
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace('Ġ', '').replace('Ċ', '')
                if clean_token and all(c in allowed for c in clean_token):
                    if token_id < len(mask):
                        mask[token_id] = 0
            
            masked_logits = logits + mask
            next_token = int(np.argmax(masked_logits))
            char = self.model._decode([next_token])
            if not any(c in allowed for c in char):
                break
            generated.append(next_token)
        return generated