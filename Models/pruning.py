import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load your data
X_train = np.load('Speakers/Features/X_train.npy')
y_train = np.load('Speakers/Features/y_train.npy')
X_val = np.load('Speakers/Features/X_val.npy')
y_val = np.load('Speakers/Features/y_val.npy')
X_test = np.load('Speakers/Features/X_test.npy')
y_test = np.load('Speakers/Features/y_test.npy')

# Ensure y arrays are 1D
y_train = y_train.ravel()
y_val = y_val.ravel()
y_test = y_test.ravel()

# Encode the labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)  # Transform only, do not fit
y_test_encoded = le.transform(y_test)  # Transform only, do not fit

# One-hot encode the labels
num_classes = len(np.unique(y_train_encoded))  # Get the actual number of unique classes
y_train_encoded = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
y_val_encoded = tf.keras.utils.to_categorical(y_val_encoded, num_classes=num_classes)
y_test_encoded = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

# Load the pre-trained model
model = tf.keras.models.load_model('Model.h5')

# Verify that the output layer of the model matches the number of classes
if model.layers[-1].output_shape[-1] != num_classes:
    model.pop()
    model.add(Dense(num_classes, activation='softmax'))

# Define the pruning parameters
pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

# Apply pruning to the model
pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)

# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define pruning callbacks
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir='./pruning_logs')
]

# Fine-tune the pruned model
pruned_model.fit(X_train, y_train_encoded, epochs=25, batch_size=32, validation_data=(X_val, y_val_encoded), callbacks=callbacks)

# Strip the pruning wrappers
final_model = sparsity.strip_pruning(pruned_model)

# Compile the final model
final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the final model
test_loss, test_accuracy = final_model.evaluate(X_test, y_test_encoded)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Save the final stripped model
final_model.save('Pruned_Stripped_Model.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_model = converter.convert()

# Save the TFLite model
with open('Pruned_Model.tflite', 'wb') as f:
    f.write(tflite_model)
