# model_info.py
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model=load_model(input("Enter Model's path"))

# Display the model summary
model.summary()

# Count the total number of trainable parameters
trainable_params = sum(
    [tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = sum(
    [tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

print(f'Total Trainable Parameters: {trainable_params}')
print(f'Total Non-Trainable Parameters: {non_trainable_params}')