import cv2
import numpy as np
import os
import mediapipe as mp
from keras.models import model_from_json
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21 * 3)
            return np.concatenate([rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

actions = np.array(['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

no_sequences = 30
sequence_length = 30

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True  # Enable experimental new converter

# Disable experimental lower tensor list ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("modelSignL.tflite", "wb") as f:
    f.write(tflite_model)


# Create a text file
with open("outputSignL.txt", "w") as f:
    f.write("TensorFlow Lite Model Information\n")
    f.write("=================================\n")
    f.write("Model Name: MyModel\n")  # Update with your actual model name
    f.write("Conversion Date: 2023-12-12\n")  # Update with the current date
    f.write("TensorFlow Version: 2.7.0\n")  # Update with your TensorFlow version
    f.write("TFLite Version: 2.7.0\n")  # Update with your TFLite version
    f.write("Conversion Details:\n")
    f.write("  - New Converter: Enabled\n")
    f.write("  - Experimental Ops: [TFLITE_BUILTINS, SELECT_TF_OPS]\n")
    f.write("  - Lower Tensor List Ops: Disabled\n")

