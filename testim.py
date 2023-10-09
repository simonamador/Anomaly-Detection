import numpy as np
import cv2

source_path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/healthy_dataset/L_view_e'
a = np.load(source_path+'/train.npy')

cv2.imshow('test',a[611])
cv2.waitKey(0)
cv2.destroyAllWindows()