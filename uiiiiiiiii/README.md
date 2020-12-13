# Main Script

## Model
 * The model weights size is very large and hence it has not been uploaded.
 * Change the model input/output dimensions of the image.

## Dataset
 * NYU Depth v2 Dataset has been used to train the model but has not bee uploaded here due to it very large size (140 GB).

## To run:
  ### To train the code
   $ python train.py
   
   * The code will download and extract the NYU dataset if it is not present. (120GB)
   * The model will then begin to train on this dataset.
   * It will create output files every 500 batches and save a model every epoch.
   
  ###
   $ python UI.py
   
   * You must have the model downloaded or must have finished training.
   * You can provide some of the test cases available or you can make your own test cases.
