import os
import shutil
from sklearn.model_selection import train_test_split
from pydub import AudioSegment
from tabulate import tabulate

# Set the path to the original dataset
original_dataset_path = 'RawData-set'

# Set the paths for the split dataset
output_dir = 'ReadyData-set'
train_dir = os.path.join(output_dir, 'Trainingdata')
val_dir = os.path.join(output_dir, 'Validationdata')
test_dir = os.path.join(output_dir, 'Testingdata')

# Create directories for training, validation, and testing sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get the list of speaker directories
speaker_dirs = [d for d in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, d))]
total_percentage=100
total_files = 0
train_files_count = 0
val_files_count = 0
test_files_count = 0

total_duration = 0
train_duration = 0
val_duration = 0
test_duration = 0

def get_audio_duration(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        duration = len(audio) / 1000  # duration in seconds
        return duration
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return 0

# Split the data for each speaker and move files to corresponding directories
for speaker_dir in speaker_dirs:
    speaker_files = os.listdir(os.path.join(original_dataset_path, speaker_dir))
    num_files = len(speaker_files)
    if num_files > 1:  # Check if there are enough files to split
        # Update total file count
        total_files += num_files

        # Split into training (60%) and test_val (40%)
        train_files, test_val_files = train_test_split(speaker_files, test_size=0.4, random_state=42)
        # Split test_val into validation (20%) and testing (20%)
        val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=42)

        # Move files to training directory
        for file in train_files:
            src = os.path.join(original_dataset_path, speaker_dir, file)
            dst = os.path.join(train_dir, speaker_dir, file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            train_files_count += 1
            train_duration += get_audio_duration(src)

        # Move files to validation directory
        for file in val_files:
            src = os.path.join(original_dataset_path, speaker_dir, file)
            dst = os.path.join(val_dir, speaker_dir, file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            val_files_count += 1
            val_duration += get_audio_duration(src)

        # Move files to testing directory
        for file in test_files:
            src = os.path.join(original_dataset_path, speaker_dir, file)
            dst = os.path.join(test_dir, speaker_dir, file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            test_files_count += 1
            test_duration += get_audio_duration(src)

# Calculate and print split percentages
train_percentage = (train_files_count / total_files) * 100
val_percentage = (val_files_count / total_files) * 100
test_percentage = (test_files_count / total_files) * 100

# Convert durations to hours for better readability
total_duration_hours = (train_duration + val_duration + test_duration) / 3600
train_duration_hours = train_duration / 3600
val_duration_hours = val_duration / 3600
test_duration_hours = test_duration / 3600

# Prepare data for the table
table_data = [
    ["Total", total_files, f"{total_percentage:.2f}%", f"{total_duration_hours:.2f} hours"],
    ["Training", train_files_count, f"{train_percentage:.2f}%", f"{train_duration_hours:.2f} hours"],
    ["Validation", val_files_count, f"{val_percentage:.2f}%", f"{val_duration_hours:.2f} hours"],
    ["Testing", test_files_count, f"{test_percentage:.2f}%", f"{test_duration_hours:.2f} hours"]
]

# Print the table
print(tabulate(table_data, headers=["Set", "Number of Files", "Percentage", "Total Duration"], tablefmt="grid"))
