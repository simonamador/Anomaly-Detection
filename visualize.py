import pandas as pd
import itertools
import matplotlib.pyplot as plt
import time
import os
import sys

models = ['default', 'residual']
views = ['L', 'A', 'S']
loss = ['L2','SSIM']
comb = [views,models,loss]
ids = list(itertools.product(*comb))

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/Results/'

res_date = str(sys.argv[1])

save_path = path+'training_curves_'+res_date

if not os.path.exists(save_path):
    os.mkdir(save_path)

for view, mode, loss in ids:
    results_path = path + view + '_' + mode + '_AE_'+loss+'_'+res_date+'/history.txt'
    if os.path.exists(results_path):
        history = pd.read_csv (results_path, header=0)

        epoch = [int(a) for a in history.iloc[:,0]]
        train_hist = [float(a) for a in history.iloc[:,1]]
        val_hist = [float(a) for a in history.iloc[:,2]]


        both_plt = plt.figure()
        plt.plot(epoch, train_hist, label='Training')
        plt.plot(epoch, val_hist, label='Validation')
        plt.legend(loc="upper left")
        plt.title('Learning curve '+view+' '+mode+' with '+loss+' loss')
        plt.xlabel('Epochs')
        plt.ylabel(loss+' Loss')

        if loss == 'SSIM':
            plt.axis([0, 1000, 0, 1])
        elif loss == 'L2':
            plt.axis([0, 1000, 0, 8000])           

        plt.savefig(save_path+'/both_'+view+'_'+mode+'_'+loss+'.png')