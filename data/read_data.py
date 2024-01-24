import os 
import sys
import csv

# Author: @simonamador

# The following code an extraction of selected healthy-subject MRI images into a local folder

# Input:
data_list = sys.argv[1] #csv file with list of processed data

# Path for extraction
source_directory = {'CHD':'/neuro/users/mri.team/fetal_mri/Data/CHD_protocol/Data/',
                    'Placenta':'/neuro/users/mri.team/fetal_mri/Data/Placenta_protocol/Data/',
                    'TMC':'/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2023/VM_study/TMC/TD/',
                    'VGH':'/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2023/VM_study/VGH/TD/',
                    'BCH':'/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2023/VM_study/final/'}

# Path for allocation of images
if data_list[:2] == 'TD':
    destination_directory = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/TD_dataset/Raw'
else:
    destination_directory = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/VM_dataset/Raw'
    source_directory['TMC'] = '/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2023/VM_study/final/'
    source_directory['VGH'] = '/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2023/VM_study/final/'   

print('/'*10)
print('Allocation of images...')

# Loop through the rows in the csv indicating thels  images to be extracted, in order to obtain the image id's to generate the paths
locations = []
with open(data_list, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        study_value = row['Study']
        id_value = row['Study ID']
        path_value = row['Path']
        locations.append((study_value, id_value, path_value))

print('Total of '+str(len(locations))+' subjects.')

# Loop through the paths to extract the images and copy them into our folder. Only the reconstruction images are copied.

for location in locations:
    # Construct source and destination paths
    study = location[0]
    folder_id = location[1]
    path = location[2]

    if study == 'VGH':
        zeros = '0'*(4-len(folder_id))
        folder_id = zeros+folder_id

    source_path = source_directory[study] + folder_id + '/' + path + '/recon_segmentation/'
    if not os.path.exists(source_path):
        source_path = source_directory[study] + folder_id + '/' + path + '/temp_recon_1/'

    if study == 'CHD' or study =='Placenta':
        folder_id += 'X'+path[-2:]
    
    os.system('cp ' + source_path + 'recon_to31_nuc.nii ' + destination_directory + '/' + folder_id + '.nii')