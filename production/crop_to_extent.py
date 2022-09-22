# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:03:49 2022

@author: tbareas
"""

from glob import glob
from PIL import Image
import os

def crop_to_ext(imagelist, box):
    # list of file names
    names = [os.path.basename(x)[14:20] for x in imagelist]  
    # list of image objects
    imgs = [Image.open(x) for x in imagelist]

    for idx, im in enumerate(imgs):
        # crop imaage on extent
        img2 = im.crop(box)
        
        # save output image
        img2.save(f'{names[idx]}_cropped.tif', dpi=(300, 300))


t34folder = '..any_path.../T34/TCIs/'     #  folder of TCI images
imgs_inlist = glob(t34folder + '*.tif')
box = (0, 0, 10854, 10854)                  # Box extent to crop to

crop_to_ext(imgs_inlist, box)

print('Crop to extent finished')







