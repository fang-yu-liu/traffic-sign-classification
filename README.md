# Traffic Sign Recognition

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project is part of the [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive) program, and some of the code are leveraged from the lecture materials.

Overview
---
This project uses deep neural networks and convolutional neural networks to classify traffic signs. The neural network model is trained using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Model accuracy is tested using online traffic sign data.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./write-up-data/training_data_class_distribution.png "Training data class distribution"
[image2]: ./write-up-data/training_data_visualization.png "Training data before preprocessing"
[image3]: ./write-up-data/training_data_visualization_after_preprocessing.png "Training data after preprocessing"
[image4]: ./samples/traffic_sign_0.jpg "Traffic Sign 1"
[image5]: ./samples/traffic_sign_1.jpg "Traffic Sign 2"
[image6]: ./samples/traffic_sign_2.jpg "Traffic Sign 3"
[image7]: ./samples/traffic_sign_3.jpg "Traffic Sign 4"
[image8]: ./samples/traffic_sign_4.jpg "Traffic Sign 5"
[image9]: ./samples/traffic_sign_5.jpg "Traffic Sign 6"
[image10]: ./write-up-data/validation_accuracy_history.png "Validation Accuracy"
[image11]: ./write-up-data/softmax_visualization.png "Softmax Visualization"

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
#### Writeup / README

##### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/fang-yu-liu/traffic-sign-classification/blob/master/Traffic_Sign_Classifier.ipynb)

#### Data Set Summary & Exploration

##### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

##### 2. Include an exploratory visualization of the dataset.

* This is the class distribution histogram for the training data:
  * Each number of class id corresponds to a given traffic sign label. (Details see [this csv file](https://github.com/fang-yu-liu/traffic-sign-classification/blob/master/signnames.csv))

![alt text][image1]

#### Design and Test a Model Architecture

##### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I use the following two methods to preprocess the image data:
* Shuffle: shuffle the data to make the neural network independent of the data order.
* Grayscale: Calculate the mean for the three color channels to make the image grayscale. Since the traffic sign feature is pretty much independent to colors.
  * RBG image data shape:  (32, 32, 3)
  * Grayscale image data shape:  (32, 32, 1)
* Normalization: perform normalization on the image data to make all data between (-1, 1)
  * Image pixel mean before normalization:  82.677589037
  * Image pixel mean after normalization:  -0.354082

Example images before preprocessing:

![alt text][image2]

Example images after preprocessing:

![alt text][image3]

##### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 				|
| Flatten	| outputs 400        									|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout	      	| keep_prob 0.5 				|
| Fully connected		| outputs 84       									|
| RELU					|												|
| Dropout	      	| keep_prob 0.5 				|
| Fully connected		| outputs 43      									|
| Softmax				|         									|           |



##### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Type of optimizer: AdamOptimizer
* Batch size: 128
* Number of epochs: 70
* Hyperparameters
 * Learning rate: 0.003
 * Probability of keeping any given unit in dropout: 0.5

##### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The summary of my final model:
* Accuracy of validation data set: 0.963
* Accuracy of test data set: 0.950

The following is a plot shows the accuracy of validation set over Epochs:

![alt text][image10]


My approaches are described below:
* The first architecture chosen was the LeNet framework from the LeNet lab. The validation accuracy was around 86%. Without any preprocessing of the data.
* Applying basic image preprocessing (shuffle, grayscale and normalization). The validation accuracy increased to 92%.
* Applying dropout technique in the full connected layers in the LeNet framework to reduce overfitting. And slightly lowering the learning rate. The validation accuracy was increased to 96%
* Still seeing some overfitting, trying to do data augmentation to further reduce the overfitting issue. However, due to GPU power limitation, the data augmentation is taking way too long. Attach the code [here](https://github.com/fang-yu-liu/traffic-sign-classification/blob/master/Traffic_Sign_Data_Augmentation.ipynb), but it's not actually used in the final training of the model.

#### Test a Model on New Images

##### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

* No entry ![alt text][image4]
* Right-of-way at the next intersection ![alt text][image5]
* Road work ![alt text][image6]
* Stop ![alt text][image8]
* Speed limit (80 km/h) ![alt text][image7]
* Roundabout mandatory ![alt text][image9]

The _Speed limit (80 km/h)_ image might be difficult to classify because it's the speed limit sign where the general feature is easy to detect (circle with something in it). But the different speed limit numbers (60, 70, 80 or 120) might be harder to distinguish.

##### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No entry      		| No entry   									|
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| Road work					| Road work											|
| Stop			| Stop      							|
| Speed limit (80 km/h)	      		|Speed limit (80 km/h)					 				|
| Roundabout mandatory	      		|Roundabout mandatory				 				|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.

##### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top 5 softmax probabilities visualizations for the six images:
![alt text][image11]

For the _Right-of-way at the next intersection_, _Road work_, _Stop_ and _No entry_ images, the model is pretty sure about the prediction. All the prediction are 100% on the correct label.

For the _Roundabout mandatory_ image, the model is relatively sure about the prediction (90%). The top five softmax probabilities for this image are:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Roundabout mandatory 									|
| .01     				| Priority road|
| .00					| Speed limit (30 km/h)												|
| .00	      			| Speed limit (100 km/h)					 				|
| .00				    | Turn right ahead 							|


For the _Speed limit (80 km/h)_, like we predicted, is the hardest one for the model. However, it still managed to get 62% guess on the 80 km/h. The top five softmax probabilities for this image are:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .62         			| Speed limit (80 km/h)   									|
| .19     				| Speed limit (30 km/h)|
| .09					| Speed limit (50 km/h)												|
| .09	      			| Speed limit (60 km/h)					 				|
| .01				    | Speed limit (80 km/h)    							|
