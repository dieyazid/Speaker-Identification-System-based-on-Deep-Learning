# quantization.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np

# Load the pruned model
pruned_model = load_model('pruned_model.h5')

# Apply quantization to the pruned model
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Save the quantized model
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_model)

# Load the dataset
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
le_classes = np.load('label_classes.npy')

# Evaluate the quantized model on testing data
interpreter = tf.lite.Interpreter(model_content=quantized_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to run inference on a single input
def run_inference(interpreter, input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Prepare the test data for the TFLite model
X_test_tflite = X_test.astype(np.float32)

# Run inference and collect predictions
y_pred = []
for i in range(len(X_test_tflite)):
    input_data = np.expand_dims(X_test_tflite[i], axis=0)
    output_data = run_inference(interpreter, input_data)
    y_pred.append(output_data)

y_pred = np.array(y_pred).squeeze()
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=le_classes))
