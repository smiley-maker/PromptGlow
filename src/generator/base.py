from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PromptGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt, max_length=120):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            temperature=0.3
        )
        print(f"Base Model Outputs: {outputs}")
        val = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Base Generate Output: {val}")
        return val