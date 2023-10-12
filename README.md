# CNN binary segmentation model, used on Sentinel 2 data.
## Sampling
This explains how the samples were taken from Sentinel 2 TCI imagery.

## Unet
This includes the samples, the UNET model and the notebook I made to train a CNN for the segmentation task. 

## Production
In the following steps I tested the model with a Sentinel TCI image.
1. Used crop_to_extent to create an equally dividable image.
2. Used split_to_tiles to split the image for the preferred input to the model.
3. Ran model.
4. Used concat tiles to concatanate the resulted predictions.
5. Finally ran georef_parsing to parse the projection and extent, same as the corresponting TCI has.

... And here is the final result, still there are some things to improve (stuff I included in howtoimprove.txt)

![result_of_segmentation](https://user-images.githubusercontent.com/113855055/193895247-1d26a7ff-115f-4dc7-bac5-b9b1936e7d74.JPG)
