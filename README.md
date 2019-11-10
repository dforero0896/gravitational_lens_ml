# gravitational_lens_ml
Repository for the gravitational lens finding challenge v2. See [here](http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html).
To set up, you can organize your data as 

```
data
	catalog 
		image_catalog2.0train.csv
	datapack2.0test
		Public
			EUC_H
			EUC_J
			EUC_VIS
			EUC_Y
	datapack2.0train  
		Public
			EUC_H
			EUC_J
			EUC_VIS
			EUC_Y
	test_multiband  # created by build_dataset.py
	train_multiband # created by build_dataset.py
```
This is easily done if you just download the dataset into the `data` folder and extract there. The `.gitignore` will ignore every file in the directory but the `README.md`.
Please find the training set [here](http://metcalf1.difa.unibo.it/DATA3/datapack2.0train.tar.gz) and the test set [here](http://metcalf1.difa.unibo.it/DATA3/datapack2.0test.tar.gz).
Although, you can run `bash src/get_dataset.sh` and it should be done for you. Mind that it could take some time.
Now you can run 
```python
python src/build_dataset.py ./
```
to build the dataset as multiband images to be used by `keras`.\
To do that, you need the `tifffile` library:
```
conda install -c conda-forge tifffile
```
or
```
pip install tifffile
```
