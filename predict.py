import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Define the custom loss function (example)
def custom_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

# Function to extract features from a single audio file
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

# Function to load label encoder
def load_label_encoder():
    le = LabelEncoder()
    le.classes_ = np.load('Model\Assets\label_classes.npy')
    return le

# Function to make a prediction on a single audio file
def predict_speaker(file_path, model, le):
    features = extract_features(file_path)
    if features is None:
        return None, None
    
    # Reshape the features for the CNN input
    features = features[np.newaxis, ..., np.newaxis]
    
    # Make a prediction
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions, axis=1)
    predicted_speaker = le.inverse_transform(predicted_label)
    
    # Get the confidence scores for all classes
    confidence_scores = predictions[0] * 100  # Convert to percentage
    
    return predicted_speaker[0], confidence_scores

# Load the pre-trained model with the custom loss function
with tf.keras.utils.custom_object_scope({'custom_loss': custom_loss}):
    model = load_model('Model/TestingModel.h5')

# Load the label encoder
le = load_label_encoder()

# Path to the audio file you want to test
test_audio_file_path = 'AudioSample\sample.wav'

# Predict the speaker
predicted_speaker, confidence_scores = predict_speaker(test_audio_file_path, model, le)
if predicted_speaker is not None:
    print(f'The predicted speaker is: {predicted_speaker}')
    print("Confidence scores for each class:")
    for speaker, score in zip(le.classes_, confidence_scores):
        print(f'{speaker}: {score:.2f}%')
else:
    print('Could not predict the speaker. Please check the audio file.')
