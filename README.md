# Internet of Things Project

## Project Description
- Run a server with bilinear convolutional neural network classify the birds
- Create an Android application to upload bird figure

## Project Structure
- ``client``: The Android application written in Flutter
- ``server``: The Flask server and the BCNN model
  - ``model``: The BCNN model
    - Dataset: The Caltech-UCSD Birds-200-2011 Dataset
    - Randomly selected 10 classes for ease of training. The class number selected are: 002, 009, 016, 020, 048, 062, 069, 073, 106, 188
    - Statistics: 
      - Iteration 234/234
      - Train Loss: 0.5125 Acc: 0.9305
      - Validation Loss: 0.1129 Acc: 0.9831
      - Training complete in 16m 6s
      - Best val accuracy: 0.983108   
  - ``main.py``: The Flask server
  - ``train.py``: The script to train the BCNN model

## Run the Project
- Train the BCNN model in the ``server`` folder:
```
$ python3 train.py
```
- Run the Flask server in the ``server`` folder by executing the command:
```
$ python3 main.py
```

## References:
- Dataset: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- BCNN model: https://github.com/cvl-umass/bilinear-cnn
- Android application: https://www.porkaone.com/2022/07/how-to-upload-images-and-display-them.html