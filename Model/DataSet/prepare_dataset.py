import os
import shutil
from sklearn.model_selection import train_test_split

# Set the path to the original dataset
original_dataset_path = 'FULL/'

# Set the paths for the split dataset
output_dir = 'SPLIT/'
train_dir = os.path.join(output_dir, 'Training data')
val_dir = os.path.join(output_dir, 'Validation data')
test_dir = os.path.join(output_dir, 'Testing data')

# Create directories for training, validation, and testing sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get the list of speaker directories
speaker_dirs = [d for d in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, d))]

# Split the data for each speaker and move files to corresponding directories
for speaker_dir in speaker_dirs:
    speaker_files = os.listdir(os.path.join(original_dataset_path, speaker_dir))
    train_files, test_val_files = train_test_split(speaker_files, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=42)
    
    # Move files to training directory
    for file in train_files:
        src = os.path.join(original_dataset_path, speaker_dir, file)
        dst = os.path.join(train_dir, speaker_dir, file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    
    # Move files to validation directory
    for file in val_files:
        src = os.path.join(original_dataset_path, speaker_dir, file)
        dst = os.path.join(val_dir, speaker_dir, file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    
    # Move files to testing directory
    for file in test_files:
        src = os.path.join(original_dataset_path, speaker_dir, file)
        dst = os.path.join(test_dir, speaker_dir, file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

# Print split percentages
total_speakers = len(speaker_dirs)
print(f"Total number of speakers: {total_speakers}")

train_speakers = len(os.listdir(train_dir))
val_speakers = len(os.listdir(val_dir))
test_speakers = len(os.listdir(test_dir))

print(f"Number of speakers in training set: {train_speakers} ({train_speakers/total_speakers*100:.2f}%)")
print(f"Number of speakers in validation set: {val_speakers} ({val_speakers/total_speakers*100:.2f}%)")
print(f"Number of speakers in testing set: {test_speakers} ({test_speakers/total_speakers*100:.2f}%)")
