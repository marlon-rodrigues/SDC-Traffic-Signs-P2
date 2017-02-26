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

[image1]: ./examples/processing-image.png "Visualization"
[image2]: ./examples/30.png "Visualization"
[image3]: ./examples/straight-or-left.png "Visualization"
[image4]: ./examples/pedestrians.png "Visualization"
[image5]: ./examples/left-aheda.png "Visualization"
[image6]: ./examples/no-entry.png "Visualization"
[image7]: ./examples/wild-animal-crossing.png "Visualization"
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
###Writeup

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

I kept it simple and I'm simply choosing a random image on the training set and plotting into the view alongside that image classification label. 

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
* validation set accuracy of ~98.5%
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

Note that, particurlaly images 3, 5, 8 and 13 might be difficult to classify. Image 3 (pedestrians) has a very peculiar shape that can be confused with many other different signs (turn left ahead, roundabouts, etc...). Image 5 (no entry) is very rough on the edges and has a blue background that might confuse the model. Image 8 (80 km/h) can be easily consuded with the sign "30 km/h", especially if you consider the shadow that is contained on the image. Finally, image 13 (120 km/h), although very clear, I was curious if the model would be able to correctly identify it as "120 km/h" and not "20 km/h".

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h      		| 30 km/h  									| 
| Go Straight or Left    			| Go Straight or Left 										|
| Pedestrians					| Roundabout mandatory											|
| Turn Left Ahead	      		| Turn Left Ahead					 				|
| No Entry			| No Entry      							|
| Wild Animals Crossing			| Wild Animals Crossing      							|
| Priority Road			| Priority Road      							|
| 80 km/h			| 50 km/h      							|
| Yeld			| Yeld      							|
| No Passing			| No Passing      							|
| Turn Right Ahead			| Turn Right Ahead      							|
| 70 km/h			| 70 km/h      							|
| 120 km/h			| 120 km/h      							|


The model was able to correctly guess 11 of the 13 traffic signs, which gives an accuracy of 84.6%. This compares favorably to the accuracy on the test set of 96.6%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the 1st image, the model is relatively sure that this is a "30 km/h" sign (probability of 1.0), and the image does contain a "30 km/h" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 30 km/h									| 
| 7.05081343e-17     				| 120 km/h 										|
| 5.96814981e-17				| 70 km/h											|
| 5.25805794e-18      			| 20 km/h					 				|
| 2.87723506e-18				    | 80 km/h      							|


For the 2nd image, the model is not so sure  that this is a "Go Straight or Left" sign (probability of 9.99999404e-01), but the image does contain a "Go Straight or Left" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99999404e-01         			| Go Straight or Left									| 
| 4.96129644e-07     				| Roundabout mandatory 										|
| 1.14460541e-07				| General caution										|
| 5.29687316e-09      			| Traffic signals					 				|
| 2.66175415e-09				    | 70 km/h      							|


For the 3rd image, the model is not so sure that this is a "Roundabout" sign (probability of 0.184), but the image contains a "Pedestrians" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.18497877         			| Roundabout mandatory									| 
| 0.12243532     				| Traffic signals 										|
| 0.09847075 				| 20 km/h										|
| 0.04620901      			| End of all speed and passing limits					 				|
| 0.04555641				    | Go straight or left      							|


For the 4rd image, the model is not so sure  that this is a "Turn Left Ahead" sign (probability of 9.99989390e-01), but the image does contain a "Turn Left Ahead" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99989390e-01         			| Turn Left Ahead								| 
| 5.77493120e-06     				| Ahead only 										|
| 1.44424644e-06				| No passing									|
| 1.02168235e-06      			| End of no passing by vehicles over 3.5 metric tons					 				|
| 6.87790703e-07				    | 120 km/h      							|


For the 5th image, the model is relatively sure that this is a "No Entry" sign (probability of 1.0), and the image does contain a "No Entry" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| No Entry							| 
| 8.30872795e-21     				| Priority road 										|
| 8.06210130e-21				| Roundabout mandatory									|
| 6.12447941e-23      			| No passing for vehicles over 3.5 metric tons					 				|
| 5.98182693e-24				    | Turn left ahead     							|


For the 6th image, the model is relatively sure that this is a "Wild Animals Crossing" sign (probability of 1.0), and the image does contain a "Wild Animals Crossing" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| Wild Animals Crossing							| 
| 1.03473810e-24     				| Double curve 										|
| 9.45632318e-25				| Slippery road									|
| 1.12905431e-29      			| Yield					 				|
| 6.10348347e-32				    | Dangerous curve to the left   							|


For the 7th image, the model is relatively sure that this is a "Priority Road" sign (probability of 1.0), and the image does contain a "Priority Road" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| Priority Road							| 
| 9.43141362e-36     				| Yield 										|
| 0.00000000e+00				| 20 km/h									|
| 0.00000000e+00      			| 30 km/h					 				|
| 0.00000000e+00				    | 50 km/h   							|


For the 8th image, the model is not so sure that this is a "50 km/h" sign (probability of 9.93587136e-01), but the image contain a "80 km/h" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.93587136e-01         			| 50 km/h							| 
| 3.46563011e-03     				| 30 km/h										|
| 1.51071395e-03				| 80 km/h									|
| 1.29817508e-03      			| 70 km/h					 				|
| 1.09184351e-04				    | 100 km/h   							|


For the 9th image, the model is very sure that this is a "Yield" sign (probability of 1.0), and the image does contain a "Yeld" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yeld							| 
| 0     				| 20 km/h										|
| 0				| 30 km/h									|
| 0     			| 50 km/h					 				|
| 0				    | 60 km/h   							|


For the 10th image, the model is relatively sure that this is a "No Passing" sign (probability of 1.0), and the image does contain a "No Passing" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| No Passing							| 
| 1.40609679e-09     				| No vehicles										|
| 4.21126023e-10				| Ahead only								|
| 2.79581046e-11      			| Vehicles over 3.5 metric tons prohibited					 				|
| 1.26290150e-11				    | No passing for vehicles over 3.5 metric tons   							|


For the 11th image, the model is very sure that this is a "Turn Right Ahead" sign (probability of 1.0), and the image does contain a "Turn Right Ahead" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Turn Right Ahead							| 
| 0     				| 20 km/h										|
| 0				| 30 km/h									|
| 0     			| 50 km/h					 				|
| 0				    | 60 km/h   							|


For the 12th image, the model is not so sure  that this is a "70 km/h" sign (probability of 9.99993682e-01), but the image does contain a "70 km/h" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99993682e-01         			| 70 km/h							| 
| 3.76005823e-06     				| 30 km/h 										|
| 1.57752686e-06				| 20 km/h									|
| 6.73609179e-07      			| 120 km/h					 				|
| 1.52378234e-07				    | 80 km/h      							|


For the 13th image, the model is not so sure  that this is a "120 km/h" sign (probability of 9.91708457e-01), but the image does contain a "120 km/h" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.91708457e-01         			| 120 km/h							| 
| 5.52987820e-03     				| 20 km/h 										|
| 1.24148233e-03				| Bicycles crossing									|
| 6.14862889e-04      			| 100 km/h					 				|
| 4.73920052e-04				    | 70 km/h      							|
