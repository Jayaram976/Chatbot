# llama_helper.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LlamaChatbot:
    def __init__(self):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32,cache_dir="/mnt/data")
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_response(self, user_input):
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        response = self.pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]["generated_text"]
        return response.split("<|assistant|>")[-1].strip()
