import os 
import csv

# Author: @simonamador

# The following code is designed to automate the processing of Ventriculomegaly MRIs using the FNNDSC's pipeline.

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/Ventriculomegaly/'
subjects = os.listdir(path+'Raw')
print(subjects)

print('-'*25)
print('Starting recon processing:')

# Loops over each subject in the directory 

for id, subject in enumerate(subjects):
    file = 'Ventriculomegaly/Raw/'+subject
    local_paths = []
    with open(file+'/quality_assessment.csv', mode ='r') as csvfile:  
        csvreader = csv.DictReader(csvfile)
        # Some of the paths in the quality assesment csv are not correct, so they must be updated.
        # The updated rows are stored in a list
        for row in csvreader:
            filename = row['filename']
            if not(filename[2]=='V'):
                filename = './Ventriculomegaly/Raw' + filename[1:]
            n_row = {'filename': filename, 
            'quality': row['quality'], 
            'slice_thickness': row['slice_thickness']}
            local_paths.append(n_row)

    # File is re-written with updated paths
    csvfile = open(file+'/quality_assessment.csv', mode='w', newline='')

    headers = ['filename', 'quality', 'slice_thickness']
    csvwriter = csv.DictWriter(csvfile,delimiter=',',fieldnames=headers)
    csvwriter.writerow(dict((heads, heads) for heads in headers)) 
    csvwriter.writerows(local_paths) 

    csvfile.close()     
    
    # Recon algorithm implemented with pipeline
    os.system('python3 /neuro/labs/grantlab/research/MRI_processing/tasmiah/script_1/auto_segmentation_v2.0.py --input_fol '+file+' --recon --alignment')

    # Reconstructed MRI is copied to its corresponding directory
    dest_path = path+'recon_img/'
    os.system('cp ' + file + '/temp_recon_1/recon_to31_nuc.nii ' + dest_path + subject + '.nii')