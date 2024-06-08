import os
import numpy as np
import librosa
import tensorflow as tf

# Load the label encoder classes
label_classes = np.load('Speakers/Features/label_classes.npy', allow_pickle=True)

# Load the trained model
model = tf.keras.models.load_model('Models\Pruned_Model.h5')

# Function to extract features from an audio segment
def extract_features(audio_segment, sample_rate, max_pad_len=174):
    try:
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=40)
        
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        mfccs = mfccs[..., np.newaxis]
        
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing audio segment\n{e}")
        return None

# Function to predict speakers from an audio segment
def predict_speakers_from_segment(audio_segment, sample_rate):
    features = extract_features(audio_segment, sample_rate)
    if features is not None:
        features = features[np.newaxis, ...]  # Add batch dimension
        predictions = model.predict(features)
        predicted_labels = label_classes[np.argmax(predictions)]
        return predicted_labels
    else:
        return "Error: Unable to extract features from the audio segment."

# Main function for testing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict speakers from an audio file.")
    parser.add_argument("audio_file", help="Path to the audio file.")
    args = parser.parse_args()

    audio_file = args.audio_file
    if os.path.exists(audio_file):
        audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
        print(sample_rate)
        predicted_speaker = predict_speakers_from_segment(audio, sample_rate)
        print(f"Predicted speaker: {predicted_speaker}")
    else:
        print("Error: Audio file not found.")
