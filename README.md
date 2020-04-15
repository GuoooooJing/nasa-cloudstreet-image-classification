NASA Cloudstreet Photos Classification
====================================

Using Convolutional Neural Network to classfiy the Cloudstreet photos provided by [NASA Impact Team](https://github.com/NASA-IMPACT/data_share).

# 1. Cloud #


### 1-1) About
This project is the CNN model to classify the cloudstreet satellite image . It consists of a two layers CNN model,
Alexnet. The criteria for performance is Accuracy. 
An output of model is a float number from 0 to 1. (0: Normal, 1: Cloudstreet)


### 1-2) Architecture
```
model/twolayer.py
model/alexnet.py
```



# 2. Dataset


### 2-1) Overview
The NASA cloudstree photos contain around 1000 for each of the "Yes" and "No" folders with not uniformed height and width. I resize each of the image to 800x800 pixel to fit my model.

```
train.ipynb
```

# 3. Train
```
train.ipynb
```

### 3-1) Optimizer 
[Adam Optimizer](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)


### 3-2) Loss Function
Cross Entropy Loss (torch.nn.CrossEntropyLoss())


### 3-3) Hyperparameter

    patch size = 64
    default learning rate = 0.001  # defalut learning ratio

    


# 4. Requirement
- [torch](http://pytorch.org/docs/master/nn.html)
- [torchvision](http://pytorch.org/docs/master/torchvision/transforms.html?highlight=torchvision%20transform)
- [pillow](https://pillow.readthedocs.io/en/stable/)
- etcs..


# 5. Usage
1) Download the image file from NASA aws server
2) To create the dataset, run the train.ipynb
3) Using the patches, train the model in train.ipynb
4) Run the last two cell for train and test

# 6. Reference
- [NASA Cloud Street Dataset](https://github.com/JooHyun-Lee/Camelyon17)
- [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)


TO-DOs
------

* Trying Resnet for better performance  
* Adding a probility heat map for image instead of the binary classification



LICENSE
-------

```
MIT License
```  
