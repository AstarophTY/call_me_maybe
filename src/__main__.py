from src.function_caller import FunctionCaller
import json
import numpy as np

test = FunctionCaller()
model = test.model

vocab_path = model.get_path_to_vocabulary_json()

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open("data/exercise_input/function_calling_tests.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

for question in questions:
    prompt = question["prompt"]
    input_ids = model._encode(prompt).tolist()[0]
    logits = model.get_logits_from_input_ids(input_ids)
    next_token_id = int(np.argmax(logits))
    predicted_word = model._decode([next_token_id])
    print(f"Prompt: {prompt}")
    print(f"Token suivant prédit: '{predicted_word}' (ID: {next_token_id})")
