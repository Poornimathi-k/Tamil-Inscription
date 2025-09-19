# Tamil Inscription Recognition Project

## Description
This project implements a hybrid deep learning framework for recognizing ancient Tamil inscriptions and mapping them to modern Tamil characters. 
It uses a combination of CNN, RNN, and attention mechanisms for robust recognition of degraded and overlapping inscriptions.
## Project Structure
├── Tamil ancient/ # Main project folder
│ ├── Modern characters/ # Folder with modern Tamil character images
│ ├── test/ # Test dataset
│ ├── train/ # Training dataset
│ ├── app.py # Application script
│ ├── cnn.ipynb # Notebook for CNN experimentation
│ ├── Full_dataset.zip # Complete dataset zip
│ ├── tamil_CNN+RNN+GNN.ipynb # Notebook for hybrid model experiments
│ ├── tamil_model.pth # Trained PyTorch model weights
│ └── tamil-anc-nn-rnn-gpu.py # Python script for hybrid model
└── dummy.py # Placeholder script
##Dataset
train/ and test/ contain annotated images for training and evaluation.
Modern characters/ contains reference images for mapping ancient characters.
Full_dataset.zip contains all images used for experiments.

