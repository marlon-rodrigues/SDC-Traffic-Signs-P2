#**Traffic Sign Recognition** 

##SDC Nanodegree - Project 2. Submitted by Marlon Rodrigues

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/processed-image.png "Visualization"
[image2]: ./examples/30.png "Visualization"
[image3]: ./examples/straight-or-left.png "Visualization"
[image4]: ./examples/pedestrians.png "Visualization"
[image5]: ./examples/left-ahead.png "Visualization"
[image6]: ./examples/no-entry.png "Visualization"
[image7]: ./examples/wild-animals-crossing.png "Visualization"
[image8]: ./examples/prioritary.png "Visualization"
[image9]: ./examples/80.png "Visualization"
[image10]: ./examples/yeld.png "Visualization"
[image11]: ./examples/no-passing.png "Visualization"
[image12]: ./examples/right-ahead.png "Visualization"
[image13]: ./examples/70.png "Visualization"
[image14]: ./examples/120.png "Visualization"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pythong and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. I kept it simple and I'm simply choosing a random image on the training set and plotting into the view alongside that image classification label. 

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images in the dataset to grayscale, as I believe that gray images can provide a better result than colored images, as colors are not as important as shapes when recognizing traffic signs. With the gray image in place, I then normalize the image using the opencv library to prevent big values from overshadow small values within each image. This process is done through a simple function that loops through each image in the dataset, convert that image to it's gray scale and finally normalize the gray image.

Here is an example of a traffic sign image before and after processing.

![alt text][image1]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training, validation and test sets is contained in the first code cell of the IPython notebook.  

The data is loaded into each one of the 3 sets simply by loading each one of the datasets we have available (train.p, valid.p and test.p)

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x9 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| DROPOUT					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 8x8x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 				|
| Fully connected					|		outputs 1024										|
| RELU					|												|
| Fully connected					|		outputs 512										|
| RELU					|												|
| Fully connected					|		outputs 256										|
| RELU					|												|
| DROPOUT					|												|
| Fully connected					|		outputs 128										|
| RELU					|												|
| Fully connected					|		outputs 43										|

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh, eight and nineth cells of the ipython notebook. 

To train the model, I first start a Tensor Flow session, then i initialize all the variables and calculate the number of images in the training set. I loop through my epochs, which is set to 50, suffle my training set and run my training operation - which consists of an optimizer of the loss operation of the softmax cross entrophy function ran across all the images in the test set. I then evaluate both the accuracy of the training and the validation set by checking the prediction of the model against the correct label.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for training the model is located in the nineth cell of the ipython notebook. 

My final model results were:
* training set accuracy of ~100%
* validation set accuracy of ~99%
* test set accuracy of 96.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I simply replicated the LeNet architecture. I chose the LeNet because is very simple and it seemed very powerful based on the exercises we did in class.
* What were some problems with the initial architecture?
Although it worked pretty well, I wasn't satisfied with the accuracy rate (~93%). 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I first added a few more convolution layers. The results improved but were still unsatisfactory. I then added dropout layers within my model in different positions until I found a position that seemed to work the best. For example, at first I added a dropout layer right after the first convolution layer, which proved very unsuccessful, as my accuracy rate dropped considerably.  
* Which parameters were tuned? How were they adjusted and why?
I mainly focused on 2 parameters: learning rate and keep_prob - for dropout. After playting around with those 2 parameters for a while I came to the solution which I believe behaves the best with my model.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I believe the most important design choices for this model are the use of convolution layers, dropouts and pooling. With the convolution layers I'm able to extract different features of the image using a pre-determined filter, which will allow the model to focus on those different features when classifying an image. Dropout layers prevents the model from overfitting - thus why I have one dropout layer within my convolution layers and another one within my fully connected layers. Finally, pooling (in this case, max pooling) helps my model focus on the most important features found in the image, which at the end it will be the ones helping the module classify an image correctly.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I decided to push the model a little further and I chose thirteen differnt images to test on it. Here are thirteen German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12] ![alt text][image13]
![alt text][image14]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
