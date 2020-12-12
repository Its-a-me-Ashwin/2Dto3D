# Main Script

## Model
 * The model is large and thus wp=ont fit on github
 * Change the model input/output dimensions of the image

## To run:
  ### To train the code
   $ python train.py
   
   * The code will download and extract the NYU dataset if it is not present. (120GB)
   * The model will then begin to train on this dataset.
   * It will create output files every 500 batches and save a model every epoch.
   
  ###
   $ python UI.py
   
   * You must have the model downloaded or finished training(500MB)
   * You can provide some of the test cases available or you can make your own test cases.
   
   
