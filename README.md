# **Traffic Sign Recognition** 
by: Harpreet Singh


The goals / steps of this project are the following:
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./histogram.jpg "Histogram of Classes"
[image2]: ./grayscale.jpg "Grayscaling and Normalizing"
[image3]: ./grayscale2.jpg "Grayscaling and Normalizing"
[image4]: ./augmented1.jpg "Example of Augmented data"
[image5]: ./augmented2.jpg "Example of Augmented data"
[image12]: ./augmentedData.jpg "Augmented Data added to classes"
[image6]: ./flatHistogram.jpg "Histogram of classes after generating new data"
[image7]: ./img0.jpg "Traffic Sign 1"
[image8]: ./img1.jpg "Traffic Sign 2"
[image9]: ./img2.jpg "Traffic Sign 3"
[image10]: ./img3.jpg "Traffic Sign 4"
[image11]: ./img4.jpg "Traffic Sign 5"



I used python to calculate summary statistics of the traffic
signs data set:

* The size of training set initially is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the 43 classes. As you can see, it is pretty skewed.

![alt text][image1]

### Model Architecture

As a first step, I decided to convert the images to grayscale because that allowed us to reduce our 3 RGB channels into a singal channel. This would allow for the neural network architecture to train much faster as it doesn't have to process three different data channels.

After that, I normalized the image data because neural networks train faster on normalized data generally. For each pixel of the image, I subtracted 128 and then divided the value by 255. Note that the pixel values were in the range [0, 255] initially.

Here are examples of a traffic sign image after grayscaling and normalizing.

![alt text][image2]
![alt text][image3]

I decided to generate additional data because some classes were heavily represented in the training set, while others weren't.
To add more data to the the data set, I used the following techniques:

I recognized some of the classes would look the same if they were flipped or rotated about one or both of its axis. However, upon implementation, this resulted in bringing my accuracy down, hence, I abandoned it quickly. 

Later on, I decided to skew the image a little and rotate it to a certain degree that didn't change the classification of the image but was still different from the original. This was followed by changing the brightness of the image slightly. I generated enough data such that there were 2010 samples for each class and the histogram was flat. 

(I consulted some blogs for the image skewing code in order to test if it helped my case)

This worked great and generated new images for a total of 80,000+ images in the training set.
However, through my testing, I found that this was more often than not overfitting the data.



Here is an example of an original image and an augmented image:

![alt text][image4]
![alt text][image5]
![alt text][image6]



My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale Normalized image   			| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 10x10x16 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten	        	| Outputs 400 				                    |
| Fully connected		| Outputs 120        							|
| RELU					|												|
| Dropout				| Keep Probability = 90%						|
| Fully connected		| Outputs 84        							|
| RELU					|												|
| Dropout				| Keep Probability = 90%						|
| Fully connected		| Outputs 43 (Classification)      				|
 


To train the model, I used an Adam Optimizer with a batch size of 128 and 30 epochs. I used a learning rate of 0.00065. These values were determined experimentally.


My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.93 
* test set accuracy of 0.908


I built my neural network architecture over the LeNet architecture but the validation accuracy was too low.

Initially, one dropout layer at the end was added and the keep_probability was experimented with. I found that 90% keep probability worked well. However, this wasn't enough to obtain the 0.93 validation accuracy, hence, I added another dropout layer earlier on.

Other values I played with were the Batch Size, Number of EPOCHS, Learning Rate, and the sigma value. The decision to increase or decrease the number of EPOCHS was made according to the loss curves plotted after each attempt. The learning rate was a big factor to prevent overfitting and underfitting. It was adjusted accordingly to the result after each attempt.

Adding dropout layers was a big and important design choice as I noticed that more often than not my model was overfitting my data and hence, I needed something to reduce the overfitting.

 

### Testing New Images


Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]

The third image might be difficult to classify because it contains the yield sign as well as half of a different sign that the network wasn't trained upon.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (60 km/hr)      		| Speed Limit (50 km/hr)   									| 
| Speed Limit (30 km/hr)      		| Speed Limit (30 km/hr)      										|
| General Caution					| General Caution											|
| Yield |  Yield					 				|
| No Entry			| No Entry      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. If you recall, the accuracy on the test set of 90.8%.

For the first image, the model is close between guessing a speed limit of 50 or 60 km/hr, and the image does not guess correctly. The top five soft max probabilities were

#### Class 2: Speed limit (50km/h) with a probability of 28.1409

Class 3: Speed limit (60km/h) with a probability of 19.895

Class 1: Speed limit (30km/h) with a probability of 6.3532

Class 11: Right-of-way at the next intersection with a probability of 1.05766

Class 25: Road work with a probability of 1.01091


For the second image, the model is faily certain that its a speed limit of 30km/hr sign, which is correct. The top five soft max probabilities were

#### Class 1: Speed limit (30km/h) with a probability of 29.7613

Class 6: End of speed limit (80km/h) with a probability of 12.1484

Class 2: Speed limit (50km/h) with a probability of 4.93994

Class 3: Speed limit (60km/h) with a probability of -2.20928

Class 0: Speed limit (20km/h) with a probability of -6.58093


For the third image, the model predicts that its a general caution sign, which is correct. The next highest guess was a Children Crossing sign which is a good guess as they are both triangular in shape. The top five soft max probabilities were

#### Class 18: General caution with a probability of 28.0687

Class 28: Children crossing with a probability of 15.8438

Class 26: Traffic signals with a probability of 10.3441

Class 11: Right-of-way at the next intersection with a probability of 8.03364

Class 27: Pedestrians with a probability of 7.64271


For the fourth image, the model is faily certain that its a yield sign, which is correct. The second best guess however, is very far from correct. Also, note that the sign in this image is to the right, rather than the center. The top five soft max probabilities were

#### Class 13: Yield with a probability of 21.5518

Class 33: Turn right ahead with a probability of 11.1613

Class 5: Speed limit (80km/h) with a probability of 11.0934

Class 15: No vehicles with a probability of 9.70679

Class 35: Ahead only with a probability of 7.61424



For the final image, the model is very certain that its a No Entry sign (probability>50%). The top five soft max probabilities were

#### Class 17: No entry with a probability of 50.7468

Class 14: Stop with a probability of 12.6044

Class 8: Speed limit (120km/h) with a probability of 4.23136

Class 9: No passing with a probability of 3.99676

Class 40: Roundabout mandatory with a probability of 1.04844


# **Update after Softmax implementation**


I implemented softmax to get the probabilities in predictions. However, my model is predicting these images incorrectly. Results anyways:


Image: 1

Class 3: Speed limit (60km/h) with a probability of 0.0701734

Class 13: Yield with a probability of 0.0514876

Class 21: Double curve with a probability of 0.0411576

Class 24: Road narrows on the right with a probability of 0.039172

Class 23: Slippery road with a probability of 0.0389406



Image: 2

Class 24: Road narrows on the right with a probability of 0.0691252

Class 3: Speed limit (60km/h) with a probability of 0.0634557

Class 13: Yield with a probability of 0.0611629

Class 10: No passing for vehicles over 3.5 metric tons with a probability of 0.0539427

Class 28: Children crossing with a probability of 0.0419378



Image: 3

Class 3: Speed limit (60km/h) with a probability of 0.0654189

Class 13: Yield with a probability of 0.0633523

Class 10: No passing for vehicles over 3.5 metric tons with a probability of 0.057465

Class 36: Go straight or right with a probability of 0.0425293

Class 11: Right-of-way at the next intersection with a probability of 0.0373378




Image: 4

Class 23: Slippery road with a probability of 0.0610215

Class 13: Yield with a probability of 0.0514501

Class 10: No passing for vehicles over 3.5 metric tons with a probability of 0.0487837

Class 40: Roundabout mandatory with a probability of 0.0422501

Class 2: Speed limit (50km/h) with a probability of 0.0398402



Image: 5

Class 10: No passing for vehicles over 3.5 metric tons with a probability of 0.0481649

Class 16: Vehicles over 3.5 metric tons prohibited with a probability of 0.0390231

Class 2: Speed limit (50km/h) with a probability of 0.0381009

Class 24: Road narrows on the right with a probability of 0.0361647

Class 23: Slippery road with a probability of 0.0357701

Thanks for reading.

