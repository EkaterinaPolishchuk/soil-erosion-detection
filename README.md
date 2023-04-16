# soil-erosion-detection
## Soil Erosion Detection

This project aims to detect soil erosion areas on satellite images using a deep learning approach. Specifically, we use the U-Net neural network architecture to train a model on a dataset of labeled images. 

# Dataset
Given:

1. Aerial photograph
2. GeoDataframe

A dataset with images and binary masks from fields of an aerial photograph was formed. Since the selected fields were of different sizes, the images and binary masks were resized to a uniform size.

The original size of the mask and image:

![image](https://github.com/EkaterinaPolishchuk/soil-erosion-detection/blob/main/data/images/23.png) ![маска](https://github.com/EkaterinaPolishchuk/soil-erosion-detection/blob/main/data/masks/23.png)

Further, data augmentation and normalization were performed before feeding into the neural network. In total, 1000 images and masks were obtained.

# Model
I use the U-Net neural network architecture to train our model on the dataset. U-Net is a convolutional neural network that is commonly used for image segmentation tasks. It consists of an encoder network that downsamples the image and a decoder network that upsamples the feature maps to produce a segmentation mask. The architecture is shown below:
![image](https://github.com/EkaterinaPolishchuk/soil-erosion-detection/blob/main/u-net-architecture.png)

# Training
I split the dataset into training, validation, and test sets. I use 90% of the dataset for training, 5% for validation, and 5% for testing. I train the model using the binary cross-entropy loss function and the Adam optimizer.

# Results
I evaluate the performance of the model on the test set using accuracy. My model achieves an accuracy score of 0.78. To improve the accuracy, it may be worth improving the quality of the images and increasing their quantity.

main.ipynb - image analysis, building a U-net model, and training results.

model_unet - saved trained model

data - part of the dataset.

requirements.txt - the required dependencies

------
Possible solutions for the task using and without using artificial intelligence are presented in [this article.](https://www.mdpi.com/2072-4292/12/24/4047). 
Pixel-based classification:
> Pixel-based image classification is widely used to replace visual analysis of image data. This approach typically involves applying decision rules to each pixel of the image. Decision rules can be automatically extracted from representative samples (supervised classification) or using iterative spectral clustering (unsupervised classification). Other approaches require expert knowledge for user-specified decision rule specification (knowledge-based classification). Brightness threshold values for highlighting erosion features are determined separately for each used orthophotomosaic based on their brightness ranges.

Object-based classification:
> Object-based image classification was developed to address the problem of fragmentation of results that arises when classifying high-resolution images based on pixels. The problem, also known as the salt-and-pepper effect, is caused by individual pixels or small groups of pixels being classified into classes different from neighboring pixels. The object-based approach solves this problem by creating spectrally homogeneous groups of pixels, called image objects or segments, followed by classifying these objects. This approach is similar to human visual interpretation. The resulting patterns are less fragmented and easier to interpret.

Object-based classification provides more realistic, larger and smoother patterns, as well as more accurately identifies transitional categories of moderately eroded soil.

The above material was taken from [this article](https://www.mdpi.com/2072-4292/12/24/4047). 
