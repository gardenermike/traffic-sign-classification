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
[image2]: ./examples/activation.png "Activation visualization"
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

I also experimented with several activation types. I used the standard [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), which did work fine (I tested with around 10 epochs with various activations to compare performance). I also used [ELU](http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf), leaky relu, and tried out self-normalizing [SELU](https://arxiv.org/abs/1706.02515) activations from a paper I found while scanning through arxiv. I noticed tht the SELU activation had the least tendency to overfit, but led to very slow training beyond around 90% accuracy. Since overfitting wasn't much a problem with dropout that I used, I went back to (leaky) ReLU. 
 

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
| High water      		| No vehicles (correct classification)   									|
| 70km/hr     			| 100km/hr 										|
| Roundabout					| Roundabout											|
| Road work	      		| Road work					 				|
| Stop			| No entry      							|
| Bumpy road | Bumpy road |
| 100km/hr | Priority road |
| Right-of-way at intersection | Right-of-way at intersection |
| Priority road | Priority road |
| Pedestrians | Pedestrians |
| Right turn ahead | Right turn ahead |
| Yield | Yield |
| No entry | No entry |

The model was able to correctly categorize 10 of the 13 traffic signs, which gives an accuracy of 77%.
The failures are all interesting.
* The 70km/hour sign was nearly predicted, but the resolution of the characters at 32x32 pixels may not have been sufficient.
* The stop sign failure suggests that a white bar on a red background is hard for the model to distinguish between stop and do not enter. Investigtion below shows that predictions were at 30% for do not enter vs 25% for stop, so this was a near-miss.
* The 100km/hr sign was just completely wrong. Since the sign was a very small portion of the 32x32 image, it appears that the model tried to look at other features in the image. Note the top-5 predictions were all below 10%. The model was lost here.

#### 3. Softmax likelihood.

Cell 31 in the notebook has code, details, and images detailing the top-5 predictions for each image. I'll provide a quick summary here of the top softmax probability for each image.


| Image			        |     Percentage	        					|
|:---------------------:|:---------------------------------------------:| 
| High water      		| 53%   									|
| 70km/hr     			| 17% 										|
| Roundabout					| 63%											|
| Road work	      		| ~100%					 				|
| Stop			| 30%      							|
| Bumpy road | 61% |
| 100km/hr | 7% |
| Right-of-way at intersection | 90% |
| Priority road | ~100% |
| Pedestrians | 21% |
| Right turn ahead | 43% |
| Yield | ~100% |
| No entry | 99% |


There aren't many surprises here. I see two strong patterns: signs making up most of the image received a much stronger predictive strength from the model. Interestingly, triangular signs seemed to fare better than round signs.
The right turn sign is an interesting outlier to me. The model was not especially confident in its prediction, even though the sign is well centered and clear to me as a human. The other predictions are all similar: left turn, roundabout, and so on. Apparently the model learned the general concept of a turn sign.
The 100km/hr sign, as discussed above, was just not found by the model at all, probably due to the small size of the sign in the image.

### Visualizing the Neural Network

![activation][image2]

To get a visualization of what the network was doing, I trained a small, quickly trained image with no pooling or strides to reduce the image size. I output the results from each filter at each layer of the image on a "no entry" sign, as I knew from my previous analysis that "no entry" signs were strongly classified.

See cell 56 (the final cell) in the notebook for a full visualization.
I expected to see a higher level of abstraction between layers, with unrecognizable features early on and strong sign signatures later on. Instead, I found that each filter in each layer responded similarly. There were subtle differences, however: some focused on the center line, some on the circle, some on the top of the center line, some the bottom, and so on.
I suspect that a deeper network would have resulted in more of the abstraction between layers that I expected, as can be seen in visualizations like [these](http://yosinski.com/deepvis).


