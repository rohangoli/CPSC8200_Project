#!/bin/bash

mkdir -p data
cd data

## Download the dataset
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

## Extract Python Dataset
tar -xvf cifar-10-python.tar.gz

## Extract Binary/C Dataset for NV-SHMem
tar -xvf cifar-10-binary.tar.gz

