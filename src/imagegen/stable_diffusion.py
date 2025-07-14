from diffusers import StableDiffusionPipeline

class ImageGenerator:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("mps")

    def generate(self, prompt):
        return self.pipe(prompt).images[0]
