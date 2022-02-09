# Landmarks Classification & Tagging using CNN 

![](assets/title_image_sydney_opera_house.jpg)

### Classify/tag a given image of a famous landmark using CNN (Convolutional Neural Network)


## Table of Contents

- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [Evaluation Criteria](#evaluation-criteria)
- [Solution Approach](#solution-approach)
- [How To Use](#how-to-use)
- [Credits](#credits)
- [License](#license)
- [Author Info](#author-info)


## Introduction
Photo sharing and photo storage services like to have location data for each uploaded photo. In addition, these services can build advanced features with the location data, such as the automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. However, although a photo's location can often be obtained by looking at the photo's metadata, many images uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the picture. However, given the large number of landmarks worldwide and the immense volume of images uploaded to photo-sharing services, using human judgment to classify these landmarks would not be feasible. In this project, we'll try to address this problem by building `Neural Network` (NN) based models to automatically predict the location of the image based on any landmarks depicted in the picture.

---
## Objective
To build NN based model that'd accept any user-supplied image as input and suggest the `top k` most relevant landmarks from '50 possible` landmarks from across the world. 

1. Download the dataset 
2. Build a CNN based neural network from scratch to classify the landmark image
   - Here, we aim to attain a test accuracy of at least 30%. At first glance, an accuracy of 30% may appear to be very low, but it's way better than random guessing, which would provide an accuracy of just 2% since we have 50 different landmarks classes in the dataset.
3. Build a CNN based neural network, using transfer-learning, to classify the landmark image
    - Here, we aim to attain a test accuracy of at least 60%, which is pretty good given the complex nature of this task.
4. Implement an inference function that will accept a file path to an image and an integer k and then predict the top k most likely landmarks this image belongs to. The print below displays the expected sample output from the predict function, indicating the top 3 (k = 3) possibilities for the image in question.

![](assets/sample_output.png)

---
## Dataset
- Dataset to be downloaded from [here](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip). Note that this is a mini dataset containing around 6,000 images); this dataset is a small subset of the [Original Landmark Dataset](https://github.com/cvdfoundation/google-landmark) that has over 700,000 images.
- Unzipped dataset would have the parent folder `landmark_images` containing training data in the `train` sub-folder and testing data in the `test` sub-folder
- There are 1250 images in the `test` sub-folder to be kept hidden and only used for model evaluation
- There are 4996 images in the `train` sub-folder to be used for training and validation
- Images in `test` and `train` sets are further categorized and kept in one of the 50 sub-folders representing 50 different landmarks classes (from 0 to 49)
- Images in the dataset are of different sizes and resolution
- Here are a few samples from the training dataset with their respective labels descriptions...

![](assets/landmark_samples.png)


---

## Evaluation Criteria

### Loss Function  
We will use `LogSoftmax` in the output layer of the network...

![](assets/LogSoftmax.png)

We need a suitable loss function that consumes these `log-probabilities` outputs and produces a total loss. The function that we are looking for is `NLLLoss` (Negative Log-Likelihood Loss). In practice, `NLLLoss` is nothing but a generalization of `BCELoss` (Binary Cross EntropyLoss or Log Loss) extended from binary-class to multi-class problem.

![](assets/NLLLoss.png)

<br>Note the `negative` sign in front `NLLLoss` formula hence negative in the name. The negative sign is put in front to make the average loss positive. Suppose we don't do this then since the `log` of a number less than 1 is negative. In that case, we will have a negative overall average loss. To reduce the loss, we need to `maximize` the loss function instead of `minimizing,` which is a much easier task mathematically than `maximizing.`


### Performance Metric

`accuracy` is used as the model's performance metric on the test-set 

![](assets/accuracy.png)

---
## Solution Approach
- Once the dataset is downloaded and unzipped, we split the training set into training and validation sets in 80%:20% (3996:1000) ratio and keep images in respective `train` and `val` sub-folders.
- `train` data is then used to build Pytorch `Dataset` object; after applying data augmentations, images are resized to 128x128.
`mean` and `standard deviation` is computed for the train dataset, and then the dataset is `normalized` using the calculated statistics. 
- The RGB channel histogram of the train set is shown below...

![](assets/train_hist1.png)

- The RGB channel histogram of the train set after normalization is shown below...

![](assets/train_hist2.png)

- Now, `test` and `val` Dataset objects are prepared in the same fashion where images are resized to 128x128 and then normalized.
- The training, validation, and testing datasets are then wrapped in Pytorch `DataLoader` object so that we can iterate through them with ease. A typical `batch_size` 32 is used.

### CNN from scratch
- The neural network is implemented as a subclass of the `nn.Module` PyTorch class. The final network presented here is built incrementally with many experiments...
    - Started with a very small CNN of just two convolutions and a linear layer with LogSoftmax output. 
    - Tried to overfit the network on a single batch of 32 training images, but the network found it hard to overfit, which means it's not powerful enough. 
    - Gradually increased the Conv and Linear layers to overfit the batch easily. 
    - Then trained on complete training data, adjusted layers, and output sizes to ensure that training loss goes down. 
    - Then, trained again with validation data to select the best network with the lowest validation loss.
    - `ReLU` is used as an activation function, and `BatchNorm` is used after every layer except the last.
    - Final model architecture (from scratch) is shown below...
    
![](assets/scratch_network.png)    

- Network initial weights are initialized by numbers drawn from a `normal-distribution in the range... 

![](assets/sqrt_n_inputs.png)

- Network is then trained and validated for 15 epochs using the `NLLLoss` function and `Adam` optimizer with a learning rate of 0.001. We save the trained model here as `ignore.pt` (ignore because we are not using it for evaluation)
- We keep track of training and validation losses. When plotted, we observe that the model starts to `overfit` very quickly.

![](assets/loss1.png)

- Now, we reset the Network initial weights to Pytorch default weight to check if there are any improvements 
- Network is then again trained and validated for 15 epochs using the `NLLLoss` function and `Adam` optimizer with a learning rate of 0.001. We save the trained model here as `model_scratch.pt` (we will use this saved model for evaluation)
- We keep track of training and validation losses. When plotted, we observe that result is almost the same as that of custom weight initialization

![](assets/loss2.png)

- The trained network (`model_scratch.pt`) is then loaded and evaluated on unseen 1,250 testing images.
The network can achieve around `38%` accuracy, which is more than we aimed for (i.e., 30%). Furthermore, the network can classify `475` images out of the total `1250` test images.

### CNN using transfer-learning
- Here, we use transfer-learning to implement the CNN network to classify images of landmarks.  
    - We have selected the `VGG19` pre-trained model on `ImageNet` as our base model.
    Models pre-trained and tested on ImgaeNet can extract general features from even the datasets that may not be very similar to ImageNet. This is due to the sheer size of the ImageNet dataset (1.2 million images) and the number of classes (1000). Instead of `VGG19`, we could have chosen `ResNet` `DenseNet`  as our base network; they would have worked just fine. `VGG19` was selected here because of its simplicity of the architecture and still producing an impressive result.
    - VGG19 models weights are frozen so that they do not change during the training. 
    - A `custom-classifier` with `ReLU` activation, `Dropouts` in hidden layers and `LogSoftmax` in last layer is created.
    The original classifier layer in VGG19 is replaced by a `custom-classifier` with learnable weights.
    - Final model architecture (transfer learning) is shown below...
    
![](assets/transfer_network.png)    

- Network is then trained and validated for ten epochs using the `NLLLoss` function and `Adam` optimizer with a learning rate of 0.001. Note that the optimizer has been supplied with the learnable parameters of `custom-classifier` only and not the whole model. This is because we want to optimize our custom-classifier weights only and use ImageNet learned weights for the rest of the layers.
- We keep track of training and validation losses and plot them. 

![](assets/loss3.png)

- The trained network is saved as `model_transfer.pt` 

- The trained network `model_transfer.pt` is then loaded and evaluated on unseen 1,250 testing images.
- This time network can achieve around `63%` accuracy, which is more than what we aimed for (i.e., 60%). In addition, the network can classify `788` images out of the total `1250` test images.
As we can see, the model built using transfer learning has outperformed the model built from scratch; hence, the second model will be used to predict unseen images.

### Interface for inference 
- For our model to be used easily, we'll implement a function `predict_landmarks` which will... 
    - Accepts a `file-path` to an image and an integer `k`
    The function expects the trained model `model_transfer.pt` to be present in the same folder/directory from where the function is invoked. The trained model can be downloaded from [here](https://drive.google.com/file/d/1c3aj2l3f3mkuH2a9orFDRNPdg0Vqa-wg/view?usp=sharing)
    - It predicts and returns the **top k most likely landmarks**. 
    - `predict_landmarks` function can be invoked from the `python` script or shell; an example is shown below...
    
    
    ```python
         >>> predicted_landmarks = predict_landmarks('images/test/09.Golden_Gate_Bridge/190f3bae17c32c37.jpg', 5)
         >>> print(predicted_landmarks)
         ['Golden Gate Bridge',
          'Forth Bridge',
          'Sydney Harbour Bridge',
          'Brooklyn Bridge',
          'Niagara Falls']
    ```
    
- We create another higher-level function, `suggest_locations` that accepts the same parameters as that of `predict_landmarks` and internally uses the `predict_landmarks` function
- A sample of function usage and its output is shown below

``` python
    >>> suggest_locations('assets/Eiffel-tower_night.jpg')
```

![](assets/eiffel_tower_prediction.png)
    
---
## How To Use
1. Ensure the below-listed packages are installed
    - `NumPy`
    - `matplotlib`
    - `torch`
    - `torchvision`
    - `cv2`
    - `PIL`
2. Download `landmark-classification-cnn-pytorch.ipynb` jupyter notebook from this repo
3. To train the models, it's recommended to execute the notebook one cell at a time. If a GPU is available (recommended), it'll use it automatically; otherwise, it'll fall back to the CPU. 
4. On a machine with `NVIDIA Quadro P5000` GPU with 16GB memory, it approximately takes 15-18 minutes to train and validate the `from scratch` model for 15 epochs
5. On a machine with `NVIDIA Quadro P5000` GPU with 16GB memory, it approximately takes 15-18 minutes to train and validate the `transfer-learning` model for ten epochs
6. A fully trained model `model_transfer.pt` can be downloaded from [here](https://drive.google.com/file/d/1c3aj2l3f3mkuH2a9orFDRNPdg0Vqa-wg/view?usp=sharing). This model then can be used directly for tagging new landmark images as described in [Interface for inference](#interface-for-inference) section. 

---
## Credits
- Dataset used in this project is provided by [Udacity](https://www.udacity.com/)
- Above dataset is a subset taken from the original landmarks dataset by Google [Original Landmark Dataset](https://github.com/cvdfoundation/google-landmark)
- Title photo by [Patty Jansen On Pixabay](https://pixabay.com/users/pattyjansen-154933/)

---
## License

MIT License

Copyright (c) [2021] [Sunil S. Singh]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the software, and to permit persons to whom the software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Author Info

- Twitter - [@_sssingh](https://twitter.com/_sssingh)
- Linkedin - [Sunil S. Singh](https://linkedin.com/in/sssingh)


---
