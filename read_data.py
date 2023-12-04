import os 
import sys
import csv

# Author: @simonamador

# The following code an extraction of selected healthy-subject MRI images into a local folder

# Input:
data_list = sys.argv[1] #csv file with list of processed data

# Path for extraction
source_directory = '/neuro/users/mri.team/fetal_mri/Data/'
# Path for allocation of images
destination_directory = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/healthy_dataset/Raw'

print('/'*10)
print('Allocation of images...')

# Loop through the rows in the csv indicating the images to be extracted, in order to obtain the image id's to generate the paths
locations = []
with open(data_list, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        study_value = row['Study']
        id_value = row['Study ID']
        path_value = row['Path']
        locations.append((study_value, id_value, path_value))
print(locations)
# Loop through the paths to extract the images and copy them into our folder. Only the reconstruction images are copied.

for location in locations:
    # Construct source and destination paths
    study = location[0] + '_protocol'
    folder_id = location[1]
    path = location[2]
    source_path = source_directory + study + '/Data/' + folder_id + '/' + path + '/recon_segmentation/'
    os.system('cp ' + source_path + 'recon_to31_nuc.nii ' + destination_directory + '/' + folder_id + '.nii')

print(source_path)
print(destination_directory)