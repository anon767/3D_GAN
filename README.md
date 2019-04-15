# Simple 3D GAN to demonstrate interpolation

## What

This jupyter notebook provides 
a super simple and intuitiv GAN implementation using Keras on Tensorflow.

It trains on the IKEA dataset particularly two models, a chair and a table.
Goal is to interpolate between two noises, one giving a artificially generated chair and another one giving a table.

The interesting part is the kinda vector algebra you could use to create mixtures of Objects like:
Chair/2 + Table/2 = mix of both

## How
Requirements:

- Python >= 3.
- Tensorflow
- Keras
- Scipy
- Jupyter notebook
- numpy
- Matplotlib

Just run through the jupyter notebook, training takes 1 hour on my laptop, resulting to a 1,5Gb large trained model.

## Literature

- http://3dgan.csail.mit.edu/