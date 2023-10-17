import os 
import sys
import csv

# Author: @simonamador

# The following code an extraction of selected healthy-subject MRI images into a local folder

# Input:
data_list = sys.argv[1] #csv file with list of processed data

# Path for extraction
source_directory = '/neuro/users/mri.team/fetal_mri/Data/Ventriculomegaly/'
# Path for allocation of images
destination_directory = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/Ventriculomegaly/Raw'

print('/'*10)
print('Allocation of images...')

# Loop through the rows in the csv indicating the images to be extracted, in order to obtain the image id's to generate the paths
locations = []
with open(data_list, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        id_value = row['subject']
        locations.append(id_value)
print(locations)
# Loop through the paths to extract the images and copy them into our folder. Only the reconstruction images are copied.

for location in locations:
    # Construct source and destination paths
    source_path = source_directory  + location 
    os.system('cp -r' + source_path + ' ' + destination_directory)

print(source_path)
print(destination_directory)