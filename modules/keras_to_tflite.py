import tensorflow as tf
from modules.loss_function import *  # Import all loss functions from the file

# Load the model with the custom loss function
model = tf.keras.models.load_model("model.keras")  # No need for custom_objects here now

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite format successfully!")
