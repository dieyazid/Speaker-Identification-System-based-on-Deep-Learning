import os
import numpy as np
import librosa
import pyaudio
import time
from sklearn.preprocessing import LabelEncoder
from predict import predict_speakers_from_segment

# Function to load label encoder
def load_label_encoder():
    le = LabelEncoder()
    le.classes_ = np.load('Speakers/Features/label_classes.npy', allow_pickle=True)
    return le

# Load the label encoder
le = load_label_encoder()

# Real-time audio processing parameters
CHUNK = int(1 * 22050)  # Number of audio samples per chunk (0.7 seconds for 16kHz sample rate)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050

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
            
            predicted_speaker = predict_speakers_from_segment(audio_segment, RATE)
            if predicted_speaker is not None:
                print(f'The predicted speaker is: {predicted_speaker}')
                
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
        
        print_speaker_intervals(speaker_intervals)

# Function to stream from an audio file
def stream_from_file(file_path):
    print(f"Playing and analyzing {file_path}...")
    audio, sample_rate = librosa.load(file_path, sr=RATE, mono=True)
    
    speaker_intervals = []
    current_speaker = None
    start_time = 0

    if len(audio) <= CHUNK:
        predicted_speaker = predict_speakers_from_segment(audio, RATE)
        if predicted_speaker is not None:
            print(f'The predicted speaker is: {predicted_speaker}')
            speaker_intervals.append((predicted_speaker, 0, len(audio) / RATE))
        else:
            print('Could not predict the speaker for the audio segment.')
    else:
        for start in range(0, len(audio), CHUNK):
            end = min(start + CHUNK, len(audio))
            audio_segment = audio[start:end]
            
            if len(audio_segment) < CHUNK:
                audio_segment = np.pad(audio_segment, (0, CHUNK - len(audio_segment)), 'constant')
            
            predicted_speaker = predict_speakers_from_segment(audio_segment, RATE)
            if predicted_speaker is not None:
                print(f'The predicted speaker is: {predicted_speaker}')
                
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
