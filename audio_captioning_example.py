# This is just an example to check whether the model works fine or not. The given audio file is taken from esc-50 and is used to know the captions predicted by the model.
from msclap import CLAP

clap_model = CLAP(version='clapcap', use_cuda=False)

captions = clap_model.generate_caption(["1-137-A-32.wav"])
print(captions)
