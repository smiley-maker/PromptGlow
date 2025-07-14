from src.generator.zero_shot import ZeroShotPromptGenerator
from src.imagegen.stable_diffusion import ImageGenerator
import matplotlib.pyplot as plt

if __name__ == "__main__":
    prompt_gen = ZeroShotPromptGenerator()
    image_gen = ImageGenerator()

    initial_query = "A sunflower "
    long_query = prompt_gen.generate(initial_query)
    print(f"Long Query: {long_query}")

#    img = image_gen.generate(long_query)
#    plt.imshow(img)
#    plt.show()
#    plt.close()