# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model (converted from ipynb)
* model.ipynb containing the scripts to visualize train data, create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments and visualizations to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 (7x7 for first layer) filter sizes and depths between 32 and 256 (model.ipynb 15 cell / model.py 230-243 lines)

I've write several helper methods to easier construct desired conv model from keras methods.

The model includes RELU layers to introduce nonlinearity (code line 193), and the data is normalized in the model using a preprocess_image function applied on all data (code lines 173-175).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 210).

I've tried to use BatchNormalization layer, but it was not so helpful.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 181). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was tuned to begin with 0.0001 and with using Keras' ReduceLROnPlateau callback (lines 256-266).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by adding compensation angles.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use simpliest deep convolution networks with tricks that I've seen that helped me on previous tasks.

My first step was to use a convolution neural network model similar to the vgg. I thought this model might be appropriate because it is pretty powerful architecture, working in many similar tasks.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I couldn't understand what is the problem, but then I've found that I'm using preprocessing function twice (one time on all images before training and one more in Keras' ImageDataGenerator) – and that was the problem.

The best way to train NN is to feed it with as many different data as you have. So I've made recordings of two laps, then I've augmented images by inverting images with angles on Y axes. Also I've used left and right cameras with compensation angles (0.25). So at final I've had almost 40k samples of data, which I've splitted 0.8/0.2 as train/valid set. Model was tested already on simulator, so there were no test set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

```python
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 64, 64, 3)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 32)        4736
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 29, 29, 64)        18496
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 29, 29, 64)        36928
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 64)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 64)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 14, 14, 128)       73856
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 14, 14, 128)       147584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 7, 128)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 7, 7, 128)         0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 7, 7, 256)         295168
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 7, 7, 256)         590080
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 3, 3, 256)         0
_________________________________________________________________
dropout_3 (Dropout)          (None, 3, 3, 256)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 2305
=================================================================
Total params: 1,169,153
Trainable params: 1,169,153
Non-trainable params: 0
```


#### 3. Creation of the Training Set & Training Process

I've just capture good enough amount of samples from two traks and augmented it by flipping on Y axes.

The best way to train NN is to feed it with as many different data as you have. So I've made recordings of two laps, then I've augmented images by inverting images with angles on Y axes. Also I've used left and right cameras with compensation angles (0.25). So at final I've had almost 40k samples of data, which I've splitted 0.8/0.2 as train/valid set. Model was tested already on simulator, so there were no test set.

Also as preprocessing step I've cropped images from simulator that there were only road left. And at the end I've resized images of the road into 64x64x3 array and normalize it by `/255. - 0.5`
