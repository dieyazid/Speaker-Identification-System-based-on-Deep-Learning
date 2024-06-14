import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Set the paths to the dataset directories
train_data_dir = 'Dataset/ReadyData-set/Training data'
val_data_dir = 'Dataset/ReadyData-set/Validation data'
test_data_dir = 'Dataset/ReadyData-set/Testing data'

# Function to load audio files and extract features
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Adjust the length of MFCC features
        if (mfccs.shape[1] > max_pad_len):
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        # Add an additional dimension for compatibility with the model
        mfccs = mfccs[..., np.newaxis]
        
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}\n{e}")
        return None

# Load data and extract features
def load_data(data_dir):
    features = []
    labels = []
    for speaker_dir in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            for file_name in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file_name)
                mfccs = extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(speaker_dir)
    return np.array(features), np.array(labels)

# Paths to save the data
asset_dir = 'Speakers/Features'
train_features_path = os.path.join(asset_dir, 'X_train.npy')
train_labels_path = os.path.join(asset_dir, 'y_train.npy')
val_features_path = os.path.join(asset_dir, 'X_val.npy')
val_labels_path = os.path.join(asset_dir, 'y_val.npy')
test_features_path = os.path.join(asset_dir, 'X_test.npy')
test_labels_path = os.path.join(asset_dir, 'y_test.npy')

# Check if preprocessed data exists
if (os.path.exists(train_features_path) and os.path.exists(train_labels_path) and
    os.path.exists(val_features_path) and os.path.exists(val_labels_path) and
    os.path.exists(test_features_path) and os.path.exists(test_labels_path)):
    
    # Load data from .npy files
    X_train = np.load(train_features_path)
    y_train = np.load(train_labels_path)
    X_val = np.load(val_features_path)
    y_val = np.load(val_labels_path)
    X_test = np.load(test_features_path)
    y_test = np.load(test_labels_path)
else:
    # Load and process data
    X_train, y_train = load_data(train_data_dir)
    X_val, y_val = load_data(val_data_dir)
    X_test, y_test = load_data(test_data_dir)

    # Check if data loading was successful
    if len(X_train) == 0 or len(y_train) == 0 or len(X_val) == 0 or len(y_val) == 0 or len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("No data found. Please check the dataset directory and file formats.")

    # Save the loaded data to .npy files
    np.save(train_features_path, X_train)
    np.save(train_labels_path, y_train)
    np.save(val_features_path, X_val)
    np.save(val_labels_path, y_val)
    np.save(test_features_path, X_test)
    np.save(test_labels_path, y_test)

# Encode the labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(le.classes_))

# Encode the labels for validation and testing sets
y_val = le.transform(y_val)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(le.classes_))

y_test = le.transform(y_test)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(le.classes_))

# Reshape data for CNN input
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Create the enhanced model
model = Sequential([
    # 1st Convolutional Block
    Conv2D(64, (3, 3), padding='same', input_shape=(40, 174, 1)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # 2nd Convolutional Block
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # 3rd Convolutional Block
    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # 4th Convolutional Block
    Conv2D(512, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Flatten and Dense Layers
    Flatten(),
    Dense(512),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    
    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    
    Dense(len(le.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save('Model.h5')

# Save label encoder classes
np.save('Speakers/Features/label_classes.npy', le.classes_)

# Evaluate the model on testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict on testing data and print classification report
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_, zero_division=1))

# Save the history to a CSV file
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# Plotting function
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Plot validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_history(history)
