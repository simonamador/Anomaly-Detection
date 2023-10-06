import os 
import sys
import csv

data_list = sys.argv[1] #csv file with list of processed data ID  & corresponding scan date

source_directory = '/neuro/users/mri.team/fetal_mri/Data/' ##replace protocol
# # Specify the destination directory
destination_directory = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/healthy_dataset/recon_img/'

print('/'*10)
print('Hola guapo ;) preparate para la copiacion')

locations = []
with open(data_list, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        study_value = row['Study']
        id_value = row['Study ID']
        path_value = row['Path']
        locations.append((study_value, id_value, path_value))

step = 0

for location in locations:
    study = location[0] + '_protocol'
    folder_id = location[1]
    path = location[2]

    # Construct source and destination paths
    source_path = source_directory + study + '/Data/' + folder_id + '/' + path + '/recon_segmentation/'
    os.system('cp ' + source_path + 'recon_to31_nuc.nii ' + destination_directory + 'image_' + str(step) + '.nii')
    step += 1
    print('todo ok ;) #perurepresent')