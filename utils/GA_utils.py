def plot_mri_slice(slice_data, title=None, cmap='gray'):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(slice_data, cmap=cmap)
    plt.axis('off')  # Remove axis ticks and labels
    if title is not None:
        plt.title(f"GA: {title:.2f}")
    plt.show()

def visualize(dl,n = 10):
    num_visualized = 0
    for batch in dl:
        images = batch['image']
        gas = batch['ga']
        for i in range(images.size(0)):
            ga_value = gas[i].item()
            if num_visualized >= n:
                return
            prCyan(f'Showing {i+1}/{n}')
            image_np = images[i, 0].squeeze().cpu().numpy()
            
            plot_mri_slice(image_np, title=ga_value)
            num_visualized += 1


if __name__ == '__main__':
    from debugging_printers import *
    from config import loader
    prGreen('Testing data adaptation strategy')
    prGreen('-'*50)
    try:
        train_dl, val_dl = loader(source_path='/neuro/labs/grantlab/research/MRI_processing/guillermo.tafoya/Anomaly_Detection/main/TD_dataset_GA/', view='S', batch_size=32, h=158)
        prGreen('Loading successful')
    except Exception as e: 
        prRed(f'Something went wrong: {e}')
    # Visualize first 10 
    visualize(train_dl)
