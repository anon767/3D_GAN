#!/usr/bin/env bash
echo "multicategorical GAN\n"
echo "Downloading pretrained model\n"
wget https://thecout.com/models/biasfree_tfbn.ckpt-18400.data-00000-of-00001
wget https://thecout.com/models/biasfree_tfbn.ckpt-18400.index
wget https://thecout.com/models/biasfree_tfbn.ckpt-18400.meta
echo "Downloading shapened"
wget http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip 

echo "moving model to correct folder"
mkdir GAN/models
mv biasfree_tfbn.ckpt-18400* GAN/models

echo "format shapes for multicategorical mode"
unzip 3DShapeNetsCode.zip
rm *.zip
mkdir 3DShapeNets/volumetric_data/multicategory_partial
cp -r 3DShapeNets/volumetric_data/airplane/* 3DShapeNets/volumetric_data/multicategorical_partial
cp -r 3DShapeNets/volumetric_data/chair/* 3DShapeNets/volumetric_data/multicategorical_partial
echo "done"
