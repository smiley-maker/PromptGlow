from src.generator.zero_shot import ZeroShotPromptGenerator
from src.generator.finetuned import FinetunedPromptGenerator
from src.imagegen.stable_diffusion import ImageGenerator
import matplotlib.pyplot as plt

if __name__ == "__main__":
    prompt_gen = FinetunedPromptGenerator()
    image_gen = ImageGenerator()

    initial_query = "An apple"
    long_query = prompt_gen.generate(initial_query)
    print(f"Long Query: {long_query}")

    oimg = image_gen.generate(initial_query)
    img = image_gen.generate(long_query)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1 row, 2 columns

    axes[0].imshow(oimg)
    axes[0].set_title('Short Prompt Result')
    axes[0].axis('off') # Hides the axes ticks and labels

    axes[1].imshow(img)
    axes[1].set_title('Long Prompt Result')
    axes[1].axis('off') # Hides the axes ticks and labels

    plt.tight_layout() # Adjusts subplot params for a tight layout
    plt.show()
