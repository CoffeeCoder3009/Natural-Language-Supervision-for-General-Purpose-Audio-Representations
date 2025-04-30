
import os
import pandas as pd
from msclap import CLAP

# Load CLAP model (Version 2023 recommended for best results)
clap_model = CLAP(version='2023', use_cuda=False)  # Set use_cuda=True if using GPU

# Path to the UrbanSound8K dataset
US8K_PATH = r"sound_datasets/urbansound8K"  # Change this if your path is different

# Load UrbanSound8K metadata
metadata = pd.read_csv(os.path.join(US8K_PATH, 'metadata', 'UrbanSound8K.csv'))

# Use a specific fold for testing (e.g., fold 10)
test_set = metadata[metadata['fold'] == 10]

# Prepare file paths for audio files
file_paths = [
    os.path.join(US8K_PATH, 'audio', f"fold{row['fold']}", row['slice_file_name'])
    for _, row in test_set.iterrows()
]

# Extract unique class labels
class_labels = list(metadata['class'].unique())

# Compute text and audio embeddings
text_embeddings = clap_model.get_text_embeddings(class_labels)
audio_embeddings = clap_model.get_audio_embeddings(file_paths)

# Compute similarities and make predictions
similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)
predictions = [class_labels[i.argmax()] for i in similarities]

# Calculate accuracy
accuracy = sum(pred == true for pred, true in zip(predictions, test_set['class'])) / len(test_set)
print(f"UrbanSound8K Test Accuracy: {accuracy * 100:.2f}%")
