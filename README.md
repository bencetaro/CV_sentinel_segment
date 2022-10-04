# CNN binary segmentation model, used on Sentinel 2 data.
## Sampling
This folder contains only the explanation how the samples were taken from Sentinel 2 TCI imagery.

## Unet
In this folder you can find the samples, the UNET model and the notebook I made to train a CNN for the segmentation task. 

## Production
Using this folders contents you can use your own certain Sentinel TCI image to test the model.
1. Use crop_to_extent to create an equally dividable image.
2. Use split_to_tiles to split the image for the preferred input to the model.
3. Run model.
4. Use concat tiles to concatanate the resulted predictions.
5. Finally run georef_parsing to parse the projection and extent, same as the corresponting TCI has.

... And here is the final result, still there are things to improve (stuff I included in howtoimprove.txt)

![result_of_segmentation](https://user-images.githubusercontent.com/113855055/193895247-1d26a7ff-115f-4dc7-bac5-b9b1936e7d74.JPG)
