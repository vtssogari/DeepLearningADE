from flairNER import train
# this is the folder in which train, test and dev files reside
data_folder = '/home/vtssogari/project/final_corps'
model_output_folder = '/home/vtssogari/project/medicine-ner'
train(data_folder, model_output_folder)
