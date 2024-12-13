# homography-deep-learning-model
This repository contains the code for training a deep learning model to predict homography transformation matrices between basketball game footage and an aerial view court. Also in this repository is the entire data reprocessing pipeline required to extract a binary mask of the basketball court at any frame of the game footage and calculate the target homography matrices that will be used for supervised training of the deep learning model. 

### Data Pre-Processing Pipeline:

The data pre-processing pipeline is governed by the [main_preprocessing.py](https://github.com/Duke-Basketball-Analytics/homography-deep-learning-model/blob/main/data_processing/scripts/main_preprocessing.py) script found in the `data_processing/scripts/` directory. Video preprocessing consists of 4 steps:

1. **Frame Extraction**: The frames are sampled from each video using a predefined frame sampling rate. This ensures there's sufficient movement of players and camera angle between observations used in training. 

2. **Augmentation Processing**: Image (frame) augmentation is performed for all frames extracted from previous step. Currently the only augmentation is resizing the images to 224x224. More augmentation such as applying kernels, binarization, or geometric transformations may be added here in the future.

3. **Mask Processing**: Each training observation (frame/image) will be paired with an additional channel representing a binary mask of the basketball court. Code for generating the court mask and saving it as a .npy file can be found in the `data_processing/court_masking` directory.

4. **Homography Processing**: Here, the target homography matrix is calculated for each frame and stored as a .npy file. Homography transformation is calculated using SIFT keypoint matching between each frame and a panoramic view of the basketball court, while a static homography transformation has been predefined between the panoramic image and an aerial view of the basketball court. The homography matrix is normalized to the Frobenius Norm such that the Frobenius Norm of all target homography matrices equals 1. This will be beneficial for implementing a Frobenius regularization constraint in the loss function. Code for generating the homography matrix can be found in the `data_processing/homography` directory.

### MSE and Reprojection Loss, with Frobenius Regularization

Ideally, a deep learning model for predicting homography matrices would utilize a reprojection loss so that gradient updates are based on the real-world application of the homography matrix. Unfortunately, homography matrices are notoriously sensitive with the slightest modification to any of the 8 degrees of freedom resulting in a completely useless reprojection and more importantly, exploding gradients. As a result, a novel approach to loss calculation is proposed where the loss function is a non-linear combination of MSE and reprojection loss. At the start of training, the reprojection component of the loss function is intentionally suppressed, leaving MSE to dominate the gradient updates. As the model begins to converge, the weight of MSE begins to decrease and the weight of reprojection begins to increase within the loss function. At the end of training, reprojection component dominates the loss function and acts as a "fine tuning". To emphasize the effect of fine-tuning, the gradial shift to reprojection loss is combined with a scheduled learning rate, understanding that the effect of reprojection loss should be gradually decreasing towards the end of training.

In addition to the MSE and reprojection loss function, an additional Frobenius regularization parameter is applied to encourage the frobenius norm of the predicted matrices to equal 1. Because homography matrices are scale invariant, we were able to normalize each matrix to a frobenius norm of 1, enabling this regularization parameter to further restrict the output of model.


