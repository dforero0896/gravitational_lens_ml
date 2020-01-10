# gravitational_lens_ml
Repository for the gravitational lens finding challenge v2. See [here](http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html).\

## Working example Setup

Do this if you don't have enough time to download and preprocess the data. You should be provided with a `data` directory containing 1000 or so examples for clipped and non clipped images (see report).\
Be sure to put the `data` directory in your `WORKDIR`, along with the `src` and `results` folders:
```
WORKDIR=path/to/your/directory/of/preference
	data
		catalog
		train_multiband_bin
		train_multiband_noclip_bin
	results
	src
```
### *PLEASE DONT FORGET TO CHANGE  THE `workdir` AND `train_multiband` ARGUMENTS IN THE CONFIGURATION FILES SINCE OUR CODE HEAVILY RELIES ON THEM TO LOCATE IMPORTANT FILES*

## Predicting on examples
We chose to provide only labeled data to decrease the file size. The `src/predict.py` script accepts two arguments: a `config_NETWORK.ini` file and a `MODEL.h5` binary. It loads the data available (at once) and predicts on it. Predictions are saved in `results/best_NETWORKpredictions.dat`.
```bash
python src/predict.py src/config_lastro.py results/best_lastro.h5
# or
python src/predict.py src/config_resnet.py results/best_resnet.h5
```
## Training on examples
For training on the provided examples, you can run either
```bash
python src/resnet.py src/config_resnet.ini
# or
python src/lastro_v1.py src/config_lastro.ini
```
from your `WORKDIR`

In our testing, `lastro_v1.py` can run in a standard GPU, while `resnet.py` asks for more memory.

## Full Setup
To set up, you can organize your data directory inside your working (`WORKDIR`) directory as 

```
data
	catalog 
		image_catalog2.0train.csv
	datapack2.0test # substructure from extracting
		Public
			EUC_H
			EUC_J
			EUC_VIS
			EUC_Y
	datapack2.0train  # substructure from extracting
		Public
			EUC_H
			EUC_J
			EUC_VIS
			EUC_Y
```
This is easily done if you just download the dataset into the `data` folder and extract there. The `.gitignore` will ignore every file in the directory but the `README.md`.
Please find the training set [here](http://metcalf1.difa.unibo.it/DATA3/datapack2.0train.tar.gz) and the test set [here](http://metcalf1.difa.unibo.it/DATA3/datapack2.0test.tar.gz). The catalog is in the 
Although, you can run `bash src/get_dataset.sh WORKDIR` and it should be done for you. Mind that it could take some time.\
Our `src` directory contains all code necessary for this implementation.\
In terms of prerequisites, everything that is needed by the project can be easily installed from our conda environment.
```
conda env create -f src/environment.yml #Assuming you are in WORKDIR
```
## Preprocessing
Now you can run from your `WORKDIR`
```python
python src/build_dataset.py ./ CLIP_PREPROCESS? OUT_DIR_NAME OVERWRITE? PARALLEL?
```
to build the dataset as multiband images to be used by `keras`. 
+ `CLIP_PRERPOCESS=1,0` defines if the built dataset is or not clipped as described in the report. 
+  `OUT_DIR_NAME (str)` is the extension to the `train_` and `test_` directory names to save the dataset. So if for instance `OUT_DIR_NAME=multiband`, the dataset will be saved in two directories: `data/train_multiband` and `data/test_multiband`. 
+  `OVERWRITE=1,0` is the flag that tells the program to or not to overwrite the files aready in place in the output dir. Finally the 
+  `PARALLEL=1,0` flag states wether you want to preprocess various files at a time. It is MPI (embarassingly) parallelized so mind you run as 
	```
	mpiexec -n PROCESSES_YOU_CAN_AFFORD python build_dataset...
	```
Even processing multiple images at a time (we did 16) it can take some hours to do the whole dataset.

## Training

We provide two `config` files, each tuned to reproduce our best results with each architecture. Be sure to edit them to add the proper paths to you working directory `workdir` and your training set directory `train_multiband` in the config files.
To create the catalog containing the labels, run
```bash
python create_labeled_catalog.py CONFIG_FILE
```
Only the `WORKDIR` is extracted from the config file so any of them will do. The script will save the catalog used by the training scripts.\
You should now be able to run
```
python src/lastro_v1.py src/config_lastro.ini
```
or
```
python src/resnet.py src/config_resnet.ini
```
to start training our best models.\
While training, the best model (with the suffix `BEST`) will be saved in the `checkpoints` directory (which is created if necessary). At the end of every epoch, the model is also saved in order to be able to resume training from the latest possible stage if needed. Regarding the latter, the model training can be stopped at any moment and resumed just by re-running the scripts. If the code finds checkpoints (or final models, saved after training has finished), those will be loaded and training will continue. To avoid this you can just change the number of epochs.

## Predicting
To  take `subsample_val` (as defined in config file) images from the testing set, loading them at once and predicting their probabilities, you can run 
```bash
python src/predict CONFIG_FILE MODEL_BINARY
```
This will save a file called `MODEL_BINARYpredictions.dat` in the `results` directory.
