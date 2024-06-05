import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pyaudio
import time

# Define the custom loss function (example)
def custom_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

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

# Function to load label encoder
def load_label_encoder():
    le = LabelEncoder()
    le.classes_ = np.load('Model\Assets\label_classes.npy')
    return le

# Function to make a prediction on an audio segment
def predict_speaker(audio_segment, sample_rate, model, le):
    features = extract_features(audio_segment, sample_rate)
    if features is None:
        return None
    
    features = features[np.newaxis, ..., np.newaxis]
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions, axis=1)
    predicted_speaker = le.inverse_transform(predicted_label)
    return predicted_speaker[0]

# Load the pre-trained model with the custom loss function
with tf.keras.utils.custom_object_scope({'custom_loss': custom_loss}):
    model = load_model('Model/TestingModel.h5')

# Load the label encoder
le = load_label_encoder()

# Real-time audio processing parameters
CHUNK = 16000  # Number of audio samples per chunk (1 second for 16kHz sample rate)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Function to stream from microphone
def stream_from_mic():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Listening for audio...")

    speaker_intervals = []
    current_speaker = None
    start_time = 0

    try:
        while True:
            data = stream.read(CHUNK)
            audio_segment = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_segment = audio_segment / np.max(np.abs(audio_segment))
            
            predicted_speaker = predict_speaker(audio_segment, RATE, model, le)
            if predicted_speaker is not None:
                print(f'The predicted speaker is: {predicted_speaker}')
                
                # Check if the speaker changed
                if current_speaker is None:
                    current_speaker = predicted_speaker
                    start_time = time.time()
                elif predicted_speaker != current_speaker:
                    end_time = time.time()
                    speaker_intervals.append((current_speaker, start_time, end_time))
                    current_speaker = predicted_speaker
                    start_time = end_time

            else:
                print('Could not predict the speaker for the current segment.')
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        end_time = time.time()
        if current_speaker is not None:
            speaker_intervals.append((current_speaker, start_time, end_time))
        print("Stopping the audio stream...")
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio stream stopped.")
        
        # Print the number of unique speakers and their intervals
        print_speaker_intervals(speaker_intervals)

# Function to stream from an audio file
def stream_from_file(file_path):
    print(f"Playing and analyzing {file_path}...")
    audio, sample_rate = librosa.load(file_path, sr=RATE, mono=True)
    
    speaker_intervals = []
    current_speaker = None
    start_time = 0

    for start in range(0, len(audio), CHUNK):
        end = min(start + CHUNK, len(audio))
        audio_segment = audio[start:end]
        
        if len(audio_segment) < CHUNK:
            audio_segment = np.pad(audio_segment, (0, CHUNK - len(audio_segment)), 'constant')
        
        predicted_speaker = predict_speaker(audio_segment, RATE, model, le)
        if predicted_speaker is not None:
            print(f'The predicted speaker is: {predicted_speaker}')
            
            # Check if the speaker changed
            if current_speaker is None:
                current_speaker = predicted_speaker
                start_time = start / RATE
            elif predicted_speaker != current_speaker:
                end_time = start / RATE
                speaker_intervals.append((current_speaker, start_time, end_time))
                current_speaker = predicted_speaker
                start_time = end_time

        else:
            print('Could not predict the speaker for the current segment.')
        
        time.sleep(1)  # Simulate real-time processing

    end_time = len(audio) / RATE
    if current_speaker is not None:
        speaker_intervals.append((current_speaker, start_time, end_time))
    
    # Print the number of unique speakers and their intervals
    print_speaker_intervals(speaker_intervals)

# Function to print the speaker intervals
def print_speaker_intervals(speaker_intervals):
    unique_speakers = set([interval[0] for interval in speaker_intervals])
    print(f"\nNumber of unique speakers: {len(unique_speakers)}")
    print("\nSpeaker Intervals:")
    print("Speaker | Start Time (s) | End Time (s)")
    print("-------------------------------------")
    for interval in speaker_intervals:
        print(f"{interval[0]} | {interval[1]:.2f} | {interval[2]:.2f}")

# Main function to choose the mode
def main():
    mode = input("Enter 'mic' to stream from microphone or 'file' to stream from an audio file: ").strip().lower()
    
    if mode == 'mic':
        stream_from_mic()
    elif mode == 'file':
        file_path = input("Enter the path to the audio file: ").strip()
        if os.path.exists(file_path):
            stream_from_file(file_path)
        else:
            print("File not found. Please check the path and try again.")
    else:
        print("Invalid option. Please enter 'mic' or 'file'.")

if __name__ == "__main__":
    main()
    print(le.classes_)
