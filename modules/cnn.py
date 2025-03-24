import numpy as np
from PIL import Image
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

def invert_image(image: Image.Image) -> Image.Image:
    img_array = np.array(image)
    inverted_array = 255 - img_array  # Invert the colors
    inverted_image = Image.fromarray(inverted_array)
    return inverted_image

def predict_image(image: Image.Image) -> Image.Image:
    img_array= np.array(image)
    # my model takes input shape ( 128, 128, 3)
    img_resized = np.resize(img_array, (128, 128, 3))
    img_resized = img_resized.astype(np.float32) / 255.0

    img_resized = np.expand_dims(img_resized, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']


    interpreter.set_tensor(input_index, img_resized)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)
    prediction = prediction.squeeze() 
    prediction = (prediction > 0.5).astype(np.uint8)

    prediction_image = Image.fromarray((prediction * 255).astype(np.uint8))
    return prediction_image