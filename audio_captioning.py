# from msclap import CLAP

# clap_model = CLAP(version='clapcap', use_cuda=False)
# captions = clap_model.generate_caption(file_paths=["1-137-A-32.wav"])
# print(captions)
from msclap import CLAP

clap_model = CLAP(version='clapcap', use_cuda=False)

# Correct parameter is `audio_paths`
captions = clap_model.generate_caption(["1-137-A-32.wav"])
print(captions)
