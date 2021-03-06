import os   
import sys
import re
import configparser
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def main():
    if len(sys.argv) == 2:
        config_file = 'config_lesta_df.ini'
        model_name = sys.argv[1]
    elif  len(sys.argv) == 3:
        config_file = sys.argv[1]
        model_name = sys.argv[2]
    else:
        sys.exit('ERROR:\tUnexpected number of arguments.\nUSAGE:\t%s [CONFIG_FILE] MODEL_FILENAME'%sys.argv[0])
    if not os.path.isfile(config_file):
        sys.exit('ERROR:\tThe config file %s was not found.'%config_file)
    if not os.path.isfile(model_name):
        sys.exit('ERROR:\tThe model file %s was not found.'%model_name)
    

    config = configparser.ConfigParser()
    config.read(config_file)
    if 'train_multiband_bin' in model_name: datadir = 'train_multiband_bin'
    elif 'train_multiband_noclip_bin' in model_name: datadir = 'train_multiband_noclip_bin'
    else: datadir = 'train_multiband_bin'
    ###### Paths
    WORKDIR = config['general']['workdir']    
    #WORKDIR = os.path.abspath(sys.argv[2])
    RESULTS = os.path.join(WORKDIR, 'results')



    bands = []
    if 'VIS0' in model_name: bands.append(False)
    elif 'VIS1' in model_name: bands.append(True)
    if 'NIR000' in model_name: [bands.append(False) for i in range(3)]
    elif 'NIR111' in model_name: [bands.append(True) for i in range(3)]
    ###### Obtain model from the saving directory
    model_name_base = os.path.basename(model_name)
    history_path =  model_name.replace('h5', 'history')
    print("Plotting %s"%model_name_base)
    ### Plots
    ## History
    if os.path.isfile(history_path):
        print("Found history file")
        with open(history_path, 'rb') as file_pi:
            history = pickle.load(file_pi)
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.plot(range(len(history['loss'])),
                history['val_loss'],
                label='Validation loss',
#                marker='o',
                c='b',
		lw=3)
        ax1.plot(range(len(history['loss'])),
                history['loss'],
                label='Training loss',
#                marker='o',
                c='r',
		lw=3)
        ax2.set_ylim([0.5,1])
        ax2.plot(range(len(history['loss'])),
                history['val_acc'],
                label='Validation accuracy',
#                marker='^',
                c='b',
                ls='--',
                fillstyle='none',
		lw=3)
        ax2.plot(range(len(history['loss'])),
                history['acc'],
                label='Training accuracy',
#                marker='^',
                c='r',
                ls='--',
                fillstyle='none',
		lw=3)
        ax1.set_xlabel('Epoch')
        ax1.legend(loc=(-0.1, 1))
        ax2.legend(loc=(0.9, 1))
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        fig.savefig(os.path.join(RESULTS, 'plots/' + os.path.basename(history_path).replace('.history', '.png')),
                    dpi=200)
        
    ##Roc curve
    roc_file = os.path.join(RESULTS, model_name_base.replace('h5', 'FPRvsTPR.dat'))
    roc_results = np.loadtxt(roc_file)
    with open(roc_file) as handler:
        header = [next(handler) for x in range(2)]
    saved_metrics = [s.split('=')[-1] for s in header]
    fpr = roc_results[:,0]
    tpr = roc_results[:,1]
    auc = float(saved_metrics[0])
    acc = float(saved_metrics[1])
    plt.figure(2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1])
    plt.legend()

    plt.plot(fpr, tpr, label='Validation\nAUC=%.3f\nACC=%.3f'%(auc, acc) ,lw =3)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], lw=3)
    plt.legend()
    plt.savefig(os.path.join(RESULTS, 'plots/ROCsklearn_' + os.path.basename(model_name).replace('.h5', '.png')),
                dpi=200)
    

if __name__ == '__main__':
    main()
