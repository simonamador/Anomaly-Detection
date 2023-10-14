import os 

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/Ventriculomegaly/'
subjects = os.listdir(path+'Raw')
print(subjects)

print('-'*25)
print('Starting recon processing:')

for id, subject in enumerate(subjects):
    file = 'Ventriculomegaly/Raw/'+subject
    os.system('python3 /neuro/labs/grantlab/research/MRI_processing/tasmiah/script_1/auto_segmentation_v2.0.py --input_fol '+file+' --recon')
    dest_path = path+'recon_img/'
    os.system('cp ' + file + '/temp_recon_1/recon_to31_nuc.nii ' + dest_path + subject + '.nii')