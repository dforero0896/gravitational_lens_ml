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
Now you can run 
```python
python src/build_dataset.py ./
```
to build the dataset as multiband images to be used by `keras`.
