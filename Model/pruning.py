# pruning.py

import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import numpy as np

# Load the student model
student_model = load_model('student_model.h5')

# Define the pruning parameters
pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                 final_sparsity=0.5,
                                                 begin_step=2000,
                                                 end_step=10000)
}

# Apply pruning to the student model
pruned_model = sparsity.prune_low_magnitude(student_model, **pruning_params)

# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the dataset
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
le_classes = np.load('label_classes.npy')

# Train the pruned model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
pruned_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the pruned model
pruned_model.save('pruned_model.h5')

# Evaluate the pruned model on testing data
test_loss, test_accuracy = pruned_model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict on testing data and print classification report
y_pred = pruned_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=le_classes))
