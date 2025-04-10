import os
import pandas as pd
from msclap import CLAP

# Load CLAP model (Version 2023 recommended for best results)
clap_model = CLAP(version='2023', use_cuda=False) # by default it uses HTSAT-22 and gpt2
# version 2022

# Load ESC-50 Dataset
# ESC50_PATH = "D:/ESC-50-master/ESC-50-master/meta"
# metadata = pd.read_csv(os.path.join(ESC50_PATH,'meta/esc-50'))
# ESC50_PATH = r"D:/ESC-50-master/ESC-50-master"
ESC50_PATH = r"D:/BTP/Datasets/ESC-50-master/ESC-50-master"

metadata = pd.read_csv(os.path.join(ESC50_PATH, 'meta', 'esc50.csv'))

# Filter Test Set (Folds 5 for testing as per ESC-50 standard)
test_set = metadata[metadata['fold'] == 4]

# Prepare File Paths and Class Labels
file_paths = [os.path.join(ESC50_PATH, 'audio', file) for file in test_set['filename']]
class_labels = list(metadata['category'].unique())

# Compute Embeddings
text_embeddings = clap_model.get_text_embeddings(class_labels)
audio_embeddings = clap_model.get_audio_embeddings(file_paths)

# Compute Accuracy
similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)
predictions = [class_labels[i.argmax()] for i in similarities]

accuracy = sum(pred == true for pred, true in zip(predictions, test_set['category'])) / len(test_set)
print(f"ESC-50 Test Accuracy: {accuracy * 100:.2f}%")
