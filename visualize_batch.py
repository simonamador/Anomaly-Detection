import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/Results/'

view = 'L'
batches = [1,8,16,32,64]
res_date = str(sys.argv[1])

save_path = path+'batches'+res_date+'/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

for batch in batches:
    batch_path = path+'L_default_AE_L2_b'+str(batch)+'_'+res_date+'/history.txt'
    if os.path.exists(batch_path):
        batch_df = pd.read_csv(batch_path,header=0)
        epoch = [int(a) for a in batch_df.iloc[:,0]]
        train_hist = [float(a) for a in batch_df.iloc[:,1]]
        val_hist = [float(a) for a in batch_df.iloc[:,2]]

        both_plt = plt.figure()
        plt.plot(epoch, train_hist, label='Training')
        plt.plot(epoch, val_hist, label='Validation')
        plt.legend(loc="upper left")
        plt.title('Learning curve with batch size '+str(batch))
        plt.xlabel('Epochs')
        plt.ylabel('L2 Loss')
        plt.axis([0, 200, 0, 8000])           

        plt.savefig(save_path+'batch_'+str(batch)+'.png')