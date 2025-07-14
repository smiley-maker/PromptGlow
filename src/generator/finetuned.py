# src/generator/finetuned.py
from .base import PromptGenerator

class FinetunedPromptGenerator(PromptGenerator):
    def __init__(self, model_dir="model_output/lora-flan-t5"):
        super().__init__(model_dir)
