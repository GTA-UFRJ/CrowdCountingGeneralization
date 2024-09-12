# CrowdCountingGeneralization
### Experiments regarding crowd Counting generalization - paper soon to be presented in SBrT 2024 XLII Simpósio Brasileiro de Telecomunicações e Processamento de Sinais

## Running the code

*There are example images and density maps (ground_truth_npy) files available in "data/mall/data". To run the code on custom data follow the steps on the end of this file. Other datasets are available in the Awesome Crowd Counting git repository: https://github.com/gjy3035/Awesome-Crowd-Counting/blob/master/src/Datasets.md*

### Steps for running the code
- Run the generate_splits.py script specifying the dataset and the number of splits: this script create N train-test splits of data and create pickle files containing lists with the image/ground-truth items in each split. (Ex.: >> python generate_splits.py mall 10) 
- Run the train_script.py script specifying the dataset and the number of splits. This step will train the model N times, where N is the number of splits. The resulting models will be saved in pickle files in the checkpoints folder. (Ex.: >> python train_script.py mall 10)
- Run the test_script.py script specifying the source dataset, the target dataset and the number of splits. The resulting MAE will be saved in a list in the results folder. (Ex.: >> python test_script.py mall mall 10) 



### Using another dataset
- Move the images of a dataset to its corresponding folder. The data folder is organized as: data/{dataset}/data/images. (Create the folders if necessary)
- Generate the density maps from ground-truth files using the "generate_density_map.py" file available in the utils folder (datasets will have different types of ground-truth files but they are generally in the .mat format)
- Density maps will be available in the data/{dataset}/data/ground_truth_npy. It may be necessary to create the folder.
- Create two folders: data/{dataset}/data/train_splits/ and data/{dataset}/data/test_splits/
- Follow the steps listed above.
  


