# Simple 3D GAN to demonstrate interpolation

## What

This jupyter notebook provides 
a super simple and intuitiv GAN implementation using Keras on Tensorflow.

It trains on the Shapenet dataset particularly two models, a chair and a airplane (17600 3D volumetric objects).
Goal is to interpolate between two noises, one giving a artificially generated chair and another one giving a airplane.

The interesting part is the kinda vector algebra you could use to create mixtures of Objects like:
Chair/2 + Airplane/2 = mix of both


## How
Requirements:

- Python >= 3.
- Tensorflow
- Scipy
- Jupyter notebook
- numpy
- Matplotlib


the shellscript downloads the pretrained models (no training needed) and the 3dshapenet data. After downloading you can check the jupyter notebook

## Literature

- http://3dgan.csail.mit.edu/
- https://github.com/rp2707/coms4995-project/tree/master/multicategory
