This is an experiment project.

It aims to compare timing between Python-numpy and C-with-Python implementations of Image Warper algorithm.

Image warper - deletes rows and columns of the image in a way that preserves most notable objects and structures. It effectively find a path of pixels from top to bottom (or left to right) that across the most "calm" areas of an image and deletes it.

So far the difference between Python-numpy and Python-C code solutions is 10x times in favor of C solution.