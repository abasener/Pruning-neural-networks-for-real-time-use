# Pruning-neural-networks-for-real-time-use
Abigail Basener
5/15/2025



# Overview: 
This project builds synthetic images from the UPWINS data. It then trains a fully connected neural network with 128 hidden nodes. Then it uses structured pruning to create 19 more models. The script will run the models on the data on the edge device, creating a results.txt file with the output values. This can then be loaded back into the Main.ipynb file to see plots of the metrics for the different models.


# How to run:
Simply place all the data, folders, and Python scripts in the correct files. Then run the Main.ipynb until the second-to-last block. Move the resalting file in the deploy_models folder to your edge device with the importOs.py file and run that script on the edge device. This will create the results.txt file with the results of the models' metrics. Move the results.txt file back onto the main hardware and run the last block in the Main.ipynb to see the plots of the metrics for the different models. 


# Needed data:
  UPWINS data repository: https://github.com/wbasener/UPWINS_Spectral_Library
  Get the following files:
    -UPWINS_4_16_2024.hdr
    -UPWINS_4_16_2024.sli


# Needed hardware:
  Main Hardware:
    -Any modern laptop or desktop with Python 3.9+

  Edge Hardware (for evaluation):
    -Any edge device you want to test that can run a Python 3.7+ script



# Required file structure:
```The UPWINS data and the main file should be in the same folder with a folder called deploy_models where the models and test data will be saved. Once importOs.py is run, the resulting results.txt should be placed in the same folder as the main file before the last block can be run.

Main Hardware:
project-root/
│
├── Main.ipynb
├── UPWINS_4_16_2024.hdr
├── UPWINS_4_16_2024.sli
├── results.txt # Created after edge run
├── deploy_models/
│ ├── y_test.pt
│ ├── x_test.pt
│ ├── vegnet_pruned_0.pt
│ ├── vegnet_pruned_0.onnx
│ ├── ...
│ ├── vegnet_pruned_99.pt
│ └── vegnet_pruned_99.onnx

Edge Hardware:
deploy_folder/
│
├── importOs.py
├── results.txt # Created after running script
├── y_test.pt
├── x_test.pt
├── vegnet_pruned_0.pt
├── vegnet_pruned_0.onnx
├── ...
├── vegnet_pruned_99.pt
└── vegnet_pruned_99.onnx
```
