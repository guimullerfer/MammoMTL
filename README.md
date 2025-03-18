
# Mammo MTL


This repository contains the source code for the study "Evaluation of Multi-Task Neural Networks to Help Diagnosis of Breast Cancer Using Mammographic Images"

## Data Organization  

To use this repository, we recommend organizing your dataset in the data folder, where 0 represents one class (e.g., normal images) and 1 represents another class (e.g., images with anomalies).

## Preprocessing

Image processing is performed by the processing function in processing.py. This function receives as parameters the image and its mask, both in grayscale.

## Train

The training pipeline is available in the train.py file. If you do not use the proposed folder system, you must change the all_cancer and all_no_cancer variables with the path of the images.