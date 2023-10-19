import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/Results/'

views = ['L', 'A', 'S']
res_date = str(sys.argv[1])

save_path = path+'GA_val_'+res_date+'/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

#plot def
width = 0.25
x = np.arange(22,40,1)

for view in views:
    ga_path = path+'GA_val_'+view+'_'+res_date+'_2.txt'
    if os.path.exists(ga_path):
        ga_e = pd.read_csv(ga_path,header=0)
        gw = [int(a) for a in ga_e.iloc[:,0]]
        l2model = [int(a) for a in ga_e.iloc[:,1]]
        ssimmodel = [int(a) for a in ga_e.iloc[:,2]]

        fig, ax = plt.subplots(layout='constrained')
        
        rects = ax.bar(x,l2model,width,label='L2')
        rects = ax.bar(x+width,ssimmodel,width,label='SSIM')

        ax.set_ylabel('L2 Loss')
        ax.set_title('Error by Gestational Age')
        ax.set_xticks(x + width, [str(g) for g in x])
        ax.legend(loc='upper left', ncols=2)
        plt.savefig(save_path+'/GA_'+view+'.png')