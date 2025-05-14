import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.silence import split_on_silence
import argparse
import keyboard

# Function to determine segment length based on vocal characteristics
def determine_segment_length(audio):
    segments = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    durations = [len(segment) / 1000 for segment in segments]  # Convert ms to seconds
    if durations:
        avg_duration = np.mean(durations)
    else:
        avg_duration = 1.0  # Default value if no segments are found
    return avg_duration

# Load the label encoder classes
label_classes = np.load('Speakers/Features/label_classes.npy', allow_pickle=True)

# Load the trained model
model = tf.keras.models.load_model('Models/Model.h5')

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
        confidence = np.max(predictions) * 100
        predicted_label = label_classes[np.argmax(predictions)]
        return predicted_label, confidence
    else:
        return "Error: Unable to extract features from the audio segment.", 0.0

# Function to merge contiguous intervals for the same speaker
def merge_intervals(intervals):
    merged_intervals = []
    if not intervals:
        return merged_intervals
    
    current_start, current_end, current_speaker, current_confidence = intervals[0]
    
    for i in range(1, len(intervals)):
        start, end, speaker, confidence = intervals[i]
        
        if speaker == current_speaker:
            current_end = end  # Extend the current interval
        else:
            merged_intervals.append((current_start, current_end, current_speaker, current_confidence))
            current_start = start
            current_end = end
            current_speaker = speaker
            current_confidence = confidence
    
    # Append the last interval
    merged_intervals.append((current_start, current_end, current_speaker, current_confidence))
    
    return merged_intervals

# Function to perform single speaker identification
def single_speaker_identification(audio_file):
    if os.path.exists(audio_file):
        try:
            audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
            predicted_label, confidence = predict_speakers_from_segment(audio, sample_rate)
            print(f"Predicted speaker: {predicted_label} with confidence: {confidence:.2f}%")
        except Exception as e:
            print(f"Error processing audio file: {e}")
    else:
        print("Error: Audio file not found.")

# Function to perform speaker diarization and generate CSV
def diarize_and_generate_csv(audio_file):
    if os.path.exists(audio_file):
        try:
            audio_pydub = AudioSegment.from_file(audio_file)
            segment_length = determine_segment_length(audio_pydub)
            audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')

            segment_samples = int(segment_length * sample_rate)

            intervals = []
            for start in range(0, len(audio), segment_samples):
                audio_segment = audio[start:start + segment_samples]
                predicted_label, confidence = predict_speakers_from_segment(audio_segment, sample_rate)
                end = min(start + segment_samples, len(audio))
                intervals.append((start / sample_rate, end / sample_rate, predicted_label, confidence))

            # Merge intervals for the same speaker
            merged_intervals = merge_intervals(intervals)

            df = pd.DataFrame(merged_intervals, columns=['Start Time (s)', 'End Time (s)', 'Speaker', 'Confidence (%)'])

            # Ensure the csv_output directory exists
            output_dir = 'csv_output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_csv = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_file))[0] + '_speaker_intervals.csv')
            df.to_csv(output_csv, index=False)
            
            print(f"Speaker intervals CSV file generated: {output_csv}")
        except Exception as e:
            print(f"Error processing audio file: {e}")
    else:
        print("Error: Audio file not found.")

# Function to record audio from the microphone
def record_audio(output_file):
    print("Recording... Press 's' to stop.")
    fs = 44100  # Sample rate
    seconds = 0  # Duration of recording, will stop on key press
    
    recording = []
    try:
        while not keyboard.is_pressed('s'):
            audio_chunk = sd.rec(int(fs * 1), samplerate=fs, channels=2, dtype='int16')
            sd.wait()  # Wait until recording is finished
            recording.append(audio_chunk)
    except KeyboardInterrupt:
        pass

    audio_data = np.concatenate(recording, axis=0)
    write(output_file, fs, audio_data)
    print(f"Recording saved to {output_file}")

# Main function with menu
def main():
    parser = argparse.ArgumentParser(description="Speaker Identification and Diarization System.")
    parser.add_argument("-m", "--mode", choices=['single', 'multi', 'record'], help="Choose mode: 'single' for single speaker identification, 'multi' for multi-speaker diarization, 'record' to record audio and identify speakers.")
    parser.add_argument("audio_file", nargs='?', help="Path to the audio file (not required for 'record' mode).")
    args = parser.parse_args()

    if args.mode == 'single' and args.audio_file:
        single_speaker_identification(args.audio_file)
    elif args.mode == 'multi' and args.audio_file:
        diarize_and_generate_csv(args.audio_file)
    elif args.mode == 'record':
        output_file = 'recorded_audio.wav'
        record_audio(output_file)
        single_speaker_identification(output_file)
    else:
        print("Invalid option or missing audio file. Use -h for help.")

if __name__ == "__main__":
    main()
