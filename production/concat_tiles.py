#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 09:51:23 2022

@author: tbareas

"""

import sys, os
from PIL import Image
from glob import glob
import re

# Function to use alphanum order
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


dir_name = "..any_path../T34TCT_TCI_try/preds/"
output_name = "T34TCT_concat_TCI.jpg"

# Read in file paths
imagelist = sorted_alphanumeric(glob(dir_name+'*.jpg'))

# Create image objects 
images = [Image.open(x) for x in imagelist]

# Take attributes
widths, heights = zip(*(i.size for i in images))

num_im =len(images)+1
num_im_per_row = 54     # in my case they are 10854x10854 images with 201x201 pixel tiles

total_width = int(sum(widths)/num_im_per_row)
total_height = int(sum(heights)/num_im_per_row)

# Create output image
new_im = Image.new('RGB', (total_width, total_height))

x_offset = 0
y_offset = 0

#seq = [x for x in range(4, num_im, 4)]   # [4, 8, 12, 16] (images 0->15 in list) // our case: [54, 108, 162..., 2916] (images 0->2917 in list)
seq = [x for x in range(num_im_per_row, num_im, num_im_per_row)]

# Concatenation process
for idx, im in enumerate(images):
    if idx in seq:
        y_offset += im.size[1]
        x_offset = 0
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
    else:    
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]

new_im.save(output_name, dpi=(300, 300))

print('Concat tiles finished')








