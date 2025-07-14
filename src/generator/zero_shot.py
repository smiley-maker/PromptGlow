# src/generator/zero_shot.py
from .base import PromptGenerator

class ZeroShotPromptGenerator(PromptGenerator):
    def __init__(self):
        super().__init__("google/flan-t5-large")

    def build_prompt(self, short_query):
        prompt = (
            f"Expand the following image generation prompt by adding art style, camera details, and scene elements. "
            f"Keep the main concept, but make it more vivid and detailed. Output at least 80 characters.\n"
            f"Prompt: {short_query}"
        )        
        return prompt
        #return f"Expand this prompt for image generation: '{short_query}'"

    def generate(self, short_query):
        prompt = self.build_prompt(short_query)
        print(f"Prompt: {prompt}")
        return super().generate(prompt)
