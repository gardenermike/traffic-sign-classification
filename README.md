# **Traffic Sign Recognition** 

---

**Applying a deep convolutional network to classify traffic signs**

The [Jupyter notebook](https://github.com/gardenermike/traffic-sign-classification/blob/master/Traffic_Sign_Classifier.ipynb) implements a deep convolutional network to classify between the 43 types of traffic signs in the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
The implementation borrows from the GoogLeNet/Inception model in that multiple convolution paths are taken, then joined together to have additional convolutions on the combined image.

Along with the model, there is also code to:
* Load the German Traffic Signs data set (see details [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset))
* Briefly explore and summarize the data
* Experiment with various activation functions and architecture. I had best performance with leaky ReLu.
* Experiment with the results of data augmentation. I found that data augmentation made a dramatic difference on generalization to images outside of the original dataset.
* Use the model to load make predictions on novel images outside of the original dataset
* Look at the top and top 5 predictions (using softmax) of the classifier on the novel images


[//]: # (Image References)

[image1]: ./examples/image_group.png "Sampling of dataset"
[image3]: ./examples/augmented_image_group.png "Augmented sample"
[image4]: ./german_traffic_signs/100.jpeg "Traffic Sign 1"
[image5]: ./german_traffic_signs/70.jpeg "Traffic Sign 2"
[image6]: ./german_traffic_signs/construction.jpeg "Traffic Sign 3"
[image7]: ./german_traffic_signs/do_not_enter.jpeg "Traffic Sign 4"
[image8]: ./german_traffic_signs/high_water.jpeg "Traffic Sign 5"
[image9]: ./german_traffic_signs/hills.jpeg "Traffic Sign 6"
[image10]: ./german_traffic_signs/kindergarten.jpeg "Traffic Sign 7"
[image11]: ./german_traffic_signs/right_turn.jpeg "Traffic Sign 8"
[image12]: ./german_traffic_signs/roundabout.jpeg "Traffic Sign 9"
[image13]: ./german_traffic_signs/stop.jpeg "Traffic Sign 10"
[image14]: ./german_traffic_signs/water_on_road.jpeg "Traffic Sign 11"
[image15]: ./german_traffic_signs/yellow_diamond.jpeg "Traffic Sign 12"
[image16]: ./german_traffic_signs/yield.jpeg "Traffic Sign 13" 

---
### Code

You can access and download an Jupyter notebook with all of the code [here](https://github.com/gardenermike/traffic-sign-classification/blob/master/Traffic_Sign_Classifier.ipynb)
Feel free to download and try it out yourself! The data will have to be downloaded separately, as it is too large to be kept inside this project. I used the preprocessed dataset provided by Udacity, which may be available [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip).

### Data Set Summary & Exploration

#### 1. A quick peek at the data:
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

* 34799 training examples
* 12630 test examples
* After processing, each image is 32x32, with 3 (RGB) channels
* There are 43 different sign types

#### 2. Sampling of dataset

Here is a random sampling of images from the dataset. Note the variety of lighting conditions and angles.

![Random sample][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### 1. Preprocessing: Normalization

In my first pass, I performed only normalization on the data. Each channel is an 8-bit integer, from 0 to 255.
To scale these values, I divided by 255 to get a value between 0 and 1, then subtracted 0.5 to center the mean around zero.
There is likely a small difference between the true mean and zero, but the difference should be negligible for training purposes.
Scaling and centering the data prevents issues with numerical stability: by keeping all of the numbers modest, rounding errors and huge gradients are less likely to pop up.
I was able to comfortably get over 90% classification accuracy without performing any additional processing, and the results are shown in the notebook.
Grayscaling would have been a possible additional processing step, but I opted not to, as color is a significant distinguishing feature between different traffic sign categories.

#### 2. Data augmentation

After my initial training, I had achieved an accuracy of over 93% on the validation set. Generalization *seemed* to be working well. However, testing on a set of images outside of the sanitized traffic sign dataset had abysmal accuracy (around 30%).
In the training dataset, all of the signs are more-or-less centered in the image, and all around the same size. Since the real world is messy, I decided to augment the data with something messier.
The Keras [ImageDataGenerator](https://keras.io/preprocessing/image/) library implements a generator that will generate batches of images with a wide range of alterations to augment original data sets. For images of animals, I would have used reflections of my data to effectively double my data set size. Since signs have meaningful non-reversible text, I did not use rotations. I added rotation (up to 10 degrees in either direction), horizontal and vertical shifting of the images (filling with nearest-neighbor in the resulting gaps), and negative zoom. Combined together, I ended up with a far more variable dataset.

Here is a sampling of the augmented data:

![augmented images][image3]

Training with the augmented images took significantly longer: I used more epochs and image processing slowed the training. In the end, I didn't get better validation set accuracy (more training here may have gradually gotten there, but I was impatient after hours and results were good enough). When testing on the novel data, however, the augmentation paid off dramatically, boosting accuracy to near 80%, and the top-5 accuracy even higher.

#### 3. Labels

I applied [one-hot encoding](https://en.wikipedia.org/wiki/One-hot) on the labels, using the [LabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) from scikit-learn.

#### 4. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Path 1: Convolution 8x8     	| 1x1 stride, same padding, outputs 32x32x8 	|
| Path 1: Activation (leaky ReLu here and all other layers)					|												|
| Path 1: Max pooling	      	| 2x2 stride,  outputs 16x16x8 				|
| Path 1: Convolution 7x7     	| 1x1 stride, same padding, outputs 16x16x16 	|
| Path 1: Activation					|												|
| Path 1: Max pooling	      	| 2x2 stride,  outputs 8x8x16 				|
| Path 1: Convolution 7x7     	| 1x1 stride, same padding, outputs 8x8x32 	|
| Path 1: Activation					|												|
| Path 2: Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x8 	|
| Path 2: Activation					|												|
| Path 2: Max pooling	      	| 2x2 stride,  outputs 16x16x8 				|
| Path 2: Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x16 	|
| Path 2: Activation					|												|
| Path 2: Max pooling	      	| 2x2 stride,  outputs 8x8x16 				|
| Path 2: Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x32 	|
| Path 2: Activation					|												|
| Path 3: Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x8 	|
| Path 3: Activation					|												|
| Path 3: Max pooling	      	| 2x2 stride,  outputs 16x16x8 				|
| Path 3: Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x16 	|
| Path 3: Activation					|												|
| Path 3: Max pooling	      	| 2x2 stride,  outputs 8x8x16 				|
| Path 3: Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x32 	|
| Path 3: Activation					|												|
| Concatenation      |   Concatenate output of paths above into a 8x8x96 output |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x128 	|
| Activation					|												|
| Dropout        | Used 50% dropout to regularize  |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| Activation					|												|
| Dropout        | Used 50% dropout to regularize  |
| Convolution 1x1     	| 1x1 stride, same padding, outputs 8x8x128 	|
| Activation					|												|
| Dropout        | Used 50% dropout to regularize  |
| Flatten        | Flatten to 8192-wide vector to send to fully connected layers |
| Fully-connected  | 256 units |
| Batch normalization  | Batch normalization at this layer resulted in less overfitting |
| Activation | |
| Dropout | 50% dropout |
| Fully-connected  | 64 units |
| Activation | |
| Dropout | 30% dropout. With the smaller layer size, I didn't want to go too heavy on dropout here. |
| Fully-connected  | 32 units |
| Activation | |
| Dropout | 20% dropout. Just a little here near the end |
| Fully-connected  | 43 units |
| Softmax activation | This is the classifier layer, so output probabilities for each class |


#### 5. Training

For the initial, non-augmented dataset, 32 epochs were sufficient to get over 93% accuracy.
For the augmented data, I needed 120 epochs to get to the same level.
I used a batch size of 32, with a learning rate of 0.0001.
I used the Adam Optimizer.
Since the Keras ImageDataGenerator runs indefinitely, I artificially created epochs based on batch size to check against the validation set. I also output training accuracy after each 25 batches.

#### 6. And... it wasn't *that* simple

My final model results were:
* training set accuracy of ~98%
* validation set accuracy of 93.3%
* test set accuracy of 93.1%

Looks great, right? As mentioned above, that wasn't my first try. I started with a LeNet-type architecture which got me up near 88%. I then played around with the ideas from GoogLeNet/Inception, which I'd been wanting to try for a while. I started with four convolutional layers in each of the three paths, but training was slow, and I found that the results weren't meaningfully better. I slowly raised my epoch count to 32 and got 93% accuracy, and thought everything was great. In particular, I noticed that I didn't have a bad overfitting problem, as my validation set accuracy tracked or exceeded the test set accuracy during most of the training. I *did* have overfitting issues on my first architecture, as the training accuracy jumped to 100% quickly, with the validation accuracy trailing behind.
Testing against novel images from the web, though, I found that my accuracy was terrible: around 30%. I had a hunch that the biggest problem was that the real images were not nicely centered. The images that *were* centered (see the "do not enter" image below) did fine.
So... I tried the data augmentation. And training was _much_ slower. I trained several times with increasing epochs, falling a bit short until I got up to 120 epochs. Training that long took about a day on an AWS p2 GPU instance (I used a spot instance to reduce the price to a reasonable level). The work paid off: the new approach generalized well to my web images.
I used dropout based on previous experience, and I didn't spend a lot of time tuning it, as I found strong resistance to overfitting with my first pass.
I suspect I could have reduced the size of my model quite a bit and still gotten pretty good results. I wanted to push it hard, though. I am going to run another pass with the epochs pushed up by 50% and see just how far I can push up the accuracy.
 

### Test a Model on New Images

#### 1. I pulled 13 images from a google image search of German traffic signs.

Here are the images:

![first image][image4] ![second image][image5] ![third image][image6] 
![fourth image][image7] ![fifth image][image8] ![sixth image][image9]
![seventh image][image10] ![eighth image][image11] ![ninth image][image12]
![tenth image][image13] ![eleventh image][image14] ![twelvth image][image15]
![thirteenth image][image16]

I tried to be a bit tricky, with duplicate/multiple signs and signs well off center in the image.

#### 2. Accuracy

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


