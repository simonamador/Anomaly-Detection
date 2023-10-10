import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

ids = [('L', 'default'),('A', 'default'),('S', 'default'),('L', 'residual'),('A', 'residual'),('S', 'residual')]

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/Results/'

res_date = str(sys.argv[1])

date = time.strftime('%Y%m%d', time.localtime(time.time()))
save_path = path+'training_curves_'+date

if not os.path.exists(save_path):
    os.mkdir(save_path)

for view, mode in ids:
    results_path = path + view + '_' + mode + '_AE_'+res_date+'/history.txt'
    history = pd.read_csv (results_path, header=0)

    epoch = [int(a) for a in history.iloc[:,0]]
    train_hist = [int(a) for a in history.iloc[:,1]]
    val_hist = [int(a) for a in history.iloc[:,2]]

    tr_plt = plt.figure()
    plt.plot(epoch, train_hist)
    plt.title('Training curve '+view+' '+mode)
    plt.xlabel('Epochs')
    plt.ylabel('L2 Loss (MSE)')
    plt.savefig(save_path+'/train_'+view+'_'+mode+'.png')

    val_plt = plt.figure()
    plt.plot(epoch, val_hist)
    plt.title('Validation curve '+view+' '+mode)
    plt.xlabel('Epochs')
    plt.ylabel('L2 Loss (MSE)')
    plt.savefig(save_path+'/val_'+view+'_'+mode+'.png')

    both_plt = plt.figure()
    plt.plot(epoch, train_hist, label='Training')
    plt.plot(epoch, val_hist, label='Validation')
    plt.legend(loc="upper left")
    plt.title('Learning curve '+view+' '+mode)
    plt.xlabel('Epochs')
    plt.ylabel('L2 Loss (MSE)')
    plt.savefig(save_path+'/both_'+view+'_'+mode+'.png')