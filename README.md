# Attempt at using Mobile Net v2.0 on DeepFashion2 dataset to detect top, bottom, and full body clothing.

Pytorch Implementation of MobileNetv2.0 is from [qfgaohao](https://github.com/qfgaohao/pytorch-ssd "Mb2 Repo") with slight adjustemnts to meet our needs. The dataset used can be found by going to [deepfashion2](https://github.com/switchablenorms/DeepFashion2 "deepfashion repo"). 

## Heirarchy of folders should be the following:
	 Root: pytorch-ssd
		->train:
			->Annotations
			->image
			...	
		->val:
			->Annotations
			->image
			...
		->deepfashion-classes.txt 
		->generate_xml_annotations.py
		->train_ssd_lite.py
		->generate_deepfashion_data.py



Ensure deepfashion2_Dataset.py is within the vision/models folder
 
## Extracting the data
```python
#create the xml annotations to be read in by generate_deepfashion_data
python generate_xml_annotations.py 'train' 
python generate_xml_annotations.py 'val' 
```
## Create the Image Sets for training
```bash
python generate_deepfashion_data.py 'train'
python generate_deepfashion_data.py 'val'
``` 
## Train the model
```bash
python train_ssd_lite.py dataset_type 'deep_fashion_2'  datasets ./  validation_dataset ./ net mb2-ssd-lite scheduler cosine lr 0.01 t_max 200 validation_epochs 5 num_epochs 20
```

