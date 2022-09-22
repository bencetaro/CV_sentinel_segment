#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:00:04 2022

@author: tbareas
"""

import os
from PIL import Image
from itertools import product


def tile(filename, dir_in, dir_out, d):
    name, _ = os.path.splitext(filename)
    ext = '.jpg'
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out, dpi=(300,300))


filename = "cropped_.tif"   # tile's output name, will be joined with the tile's position in the name
dir_in = "...any_path.../T34/arable_mask/cropped/"  
dir_out = "....any_path..../cropped/T34TCT_TCI/"
d = 201     # 54x54 tile pieces with 201x201 pixels each if original size 10854x10854

tile(filename,dir_in,dir_out,d)

print("Split to tiles finished")

