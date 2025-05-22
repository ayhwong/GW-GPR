#!/bin/bash

# use zenodo_get to pull the model
pip install zenodo_get
zenodo_get 10.5281/zenodo.15490120
unzip model.zip

# remove intermediate files
rm -f model.zip
rm -f md5sums.txt

