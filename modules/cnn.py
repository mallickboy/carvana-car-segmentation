import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

class model:
    def __init__(self, path):
        self.model = load_model(path, compile=False)

    def invert_image(self, image: Image.Image) -> Image.Image:
        img_array = np.array(image)
        inverted_array = 255 - img_array  # invert the colors
        inverted_image = Image.fromarray(inverted_array)
        return inverted_image

    def predict_image(self, image: Image.Image) -> Image.Image:     # TESTED
        img_array= np.array(image)

        # my model takes input shape ( 128, 128, 3)
        img_resized = Image.fromarray(img_array).resize((128, 128)) # using pil to resize the image
        
        # plt.imshow(img_resized)
        # plt.show()

        img_resized = np.array(img_resized).astype(np.float32) / 255.0
        img_resized = np.expand_dims(img_resized, axis=0) # shape (1, 128, 128, 3)) as it takes batch of images

        prediction = self.model.predict(img_resized)
        prediction = prediction.squeeze() 
        prediction = (prediction > 0.5).astype(np.uint8)

        return Image.fromarray((prediction * 255).astype(np.uint8))
