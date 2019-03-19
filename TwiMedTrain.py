from jsonCorps2conll03 import rmdir
from flairNER import train

# this is the folder in which train, test and dev files reside
conll_03_corps_folder = 'twimed_conll_03'
model_output_folder = 'twimed-ner'
rmdir(conll_03_corps_folder)
train(conll_03_corps_folder, model_output_folder)