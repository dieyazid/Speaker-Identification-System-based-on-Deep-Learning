# optimize.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the pre-trained teacher model
teacher_model = load_model('TestingModel.h5')

# Load the dataset
X_train = np.load('Assets\X_train.npy')
y_train = np.load('Assets\y_train.npy')
X_val = np.load('Assets\X_val.npy')
y_val = np.load('Assets\y_val.npy')
X_test = np.load('Assets\X_test.npy')
y_test = np.load('Assets\y_test.npy')
le_classes = np.load('Assets\label_classes.npy')

# Encode labels to numerical values if they are not already encoded
le = LabelEncoder()
le.classes_ = le_classes  # Restore the label encoder classes

# Ensure that y_train, y_val, and y_test are numerical and one-hot encoded
y_train = le.transform(y_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(le_classes))
y_val = le.transform(y_val)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(le_classes))
y_test = le.transform(y_test)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(le_classes))

# Define the student model
student_model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(40, 174, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le_classes), activation='softmax')
])

# Compile the student model
student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to perform knowledge distillation
def knowledge_distillation_loss(y_true, y_pred, temperature=5):
    y_true_soft = tf.keras.backend.softmax(y_true / temperature)
    y_pred_soft = tf.keras.backend.softmax(y_pred / temperature)
    return tf.keras.losses.KLDivergence()(y_true_soft, y_pred_soft) * (temperature ** 2)

# Training the student model using the teacher model's outputs
def train_student_model(teacher_model, student_model, X_train, y_train, X_val, y_val, temperature=5, alpha=0.5):
    # Generate teacher model predictions
    teacher_predictions = teacher_model.predict(X_train)
    
    # Define the custom loss
    def custom_loss(y_true, y_pred):
        return alpha * knowledge_distillation_loss(y_true, y_pred, temperature) + (1 - alpha) * tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    student_model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    student_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_val, y_val))

train_student_model(teacher_model, student_model, X_train, y_train, X_val, y_val)

# Save the student model
student_model.save('student_model.h5')

# Evaluate the student model on testing data
test_loss, test_accuracy = student_model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict on testing data and print classification report
y_pred = student_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=le_classes))
