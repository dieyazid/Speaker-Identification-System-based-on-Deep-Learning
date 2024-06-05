import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Set the paths to the dataset directories
train_data_dir = 'DataSet\SPLIT\Training data'
val_data_dir = 'DataSet\SPLIT\Validation data'
test_data_dir = 'DataSet\SPLIT\Testing data'

# Function to load audio files and extract features
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Adjust the length of MFCC features
        if mfccs.shape[1] > max_pad_len:
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

# Load training data
X_train, y_train = load_data(train_data_dir)

# Load validation data
X_val, y_val = load_data(val_data_dir)

# Load testing data
X_test, y_test = load_data(test_data_dir)

# Check if data loading was successful
if len(X_train) == 0 or len(y_train) == 0 or len(X_val) == 0 or len(y_val) == 0 or len(X_test) == 0 or len(y_test) == 0:
    raise ValueError("No data found. Please check the dataset directory and file formats.")

# Save the loaded data to .npy files
np.save('Assets\X_train.npy', X_train)
np.save('Assets\y_train.npy', y_train)
np.save('Assets\X_val.npy', X_val)
np.save('Assets\y_val.npy', y_val)
np.save('Assets\X_test.npy', X_test)
np.save('Assets\y_test.npy', y_test)

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

# Create the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, 174, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save('TestingModel.h5')

# Save label encoder classes
np.save('label_classes.npy', le.classes_)

# Evaluate the model on testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict on testing data and print classification report
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))
