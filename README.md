# Symmetry-Aware-Autoencoding

This repository provides codes for performing s-PCA and s-nlPCA outlined in the paper 'Symmetry-Aware Autoencoders: s-PCA and s-nlPCA (arxiv link).

#-----------------------------------------

#Structure

Each of the scripts belonging to a dataset (Burgers_Equation, Sudden_Expansion, Kolmogorov_Flow) can be downloaded and run independently from each other.
Each of these holdsneeds access to the directory called DATA to which the corresponding data from our repository (https://zenodo.org/record/5642671) should be downloaded. Each directory also contains
a subdirectory called models, where the scripts save the trained models.

#-----------------------------------------

#Burgers' Equation

Contained within are the scripts for calculating PCA (standard eigenvalue problem), s-PCA (NN followed by eigenvalue problem), nlPCA (NN) and s-nlPCA (NN).
Due to the simplicity of the problem there are no seperate training and evaluation scripts, but rather the evaluation is carried out right after the training.

#-----------------------------------------

#Sudden Expansion

Contains scripts for all presented methods. Due to the larger datasize the symmetry-aware models are split into training and plotting scripts. 

#-----------------------------------------

#Kolmogorov Flow

Contains scripts for all presented methods. Due to the larger datasize the symmetry-aware models are split into training and plotting scripts. 

#-----------------------------------------

#-----------------------------------------

#Prerequisities

Earlier (or later for tensorflow) versions might work but have not been tested

Python 3.6.9

  numpy       1.18.5
  
  tensorflow  2.3.1 
  
  matplotlib  3.3.3
  
  scipy       1.5.4


