# Data Science Portfolio

This portfolio is a compilation of selected notebooks which I created while exploring Machine Learning and Deep Learning landscape. 


### 1.  [Face Recognition using FaceNet](https://github.com/puneet-08/projects/blob/main/FaceNet.ipynb)

Implemented a model to identify the celebrity in an input image. I used a pretrained model provided by [Hiroki Taniai](https://github.com/nyoki-mtl/keras-facenet) trained on [MS-Celeb-1M dataset](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/). Using this model, I was able to successfuly identify individuals present in the images taken from a smaller dataset [5 Celebrity Dataset](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset).

Reference Research Paper: [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).


### 2.  [Image Segmentation using U-Net](https://github.com/puneet-08/projects/blob/main/Image_Segmentation_U_Net.ipynb)

Implemented U-Net architecture to perform the task of image segmentation on images containing neuclei. Each image in the training dataset contained a group of neuclei and associated mask. The task was to identify each of the neucleus present in the images and hence predict a correponding mask. The dataset was taken from [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018/data).

Reference Research Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

### 3.  [Neural Style Transfer](https://github.com/puneet-08/projects/blob/main/Neural_Style_Transfer_Book.ipynb) 

Implemented the technique of Neural Style transfer to generate artwork using Convolutional Neural Networks. 

Reference Research Paper: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).


### 4.  [Object Detection using YOLO-v3](https://github.com/puneet-08/projects/blob/main/YOLO_v3_Object_Detection.ipynb)

Used a pretrained model trained on [MS-COCO](https://cocodataset.org/#home) dataset to perform the task of object detection. Weights and Model architecture is provided by [Joseph Redmon](https://pjreddie.com/darknet/yolo/) - one of the inventors.

Reference Research Paper:[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)


### 5.  [Neural Machine Translation with Attention Mechanism](https://github.com/puneet-08/projects/blob/main/NMT_%2B_Attention.ipynb)

Used Seq2Seq encoder - decoder architecture to translate spanish text to english text. The model is trained on selected sentence pairs of Tatoeba Project curated by [ManyThings](http://www.manythings.org/anki/).

Reference Research Paper: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)


### 6.  [Generating Image Captions with Attention Mechanism](https://github.com/puneet-08/projects/blob/main/Image_Captioning_with_Attention.ipynb) 

Used encoder decoder architecture, where, encoder is based on CNN architecture and decoder is based on RNN architecture. The model is trained on [MS-COCO](https://cocodataset.org/#home) dataset using 'imagenet' weights.

Reference Research Paper: [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)

### 7. [House Prices - Advanced Regression Techniques](https://www.kaggle.com/startpuneet/ames-housing-prices-predictions#Housing-Price-Prediction)

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) is a knowledge competition on Kaggle. This is a regression problem: based on information about houses we predict their prices. General description and data are available on Kaggle. 

### 8. Titanic - Machine Learning from Disaster

Like many others, I also started my data science journey from this [knowledge competition](https://www.kaggle.com/c/titanic) hosted on Kaggle. This is a classification problem: aim is to predict whether a person survives or not based on some input features.

### 9. Coding ML algorithms from scratch
* [Gradient Descent and Newton's Method](https://github.com/puneet-08/projects/blob/main/Gradient_Descent_%2B_Newton's_Method.ipynb)
* [Linear Regression](https://github.com/puneet-08/projects/blob/main/Linear_Regression_from_scratch.ipynb)
* [Logistic Regression](https://github.com/puneet-08/projects/blob/main/Logistic_Regression_from_scratch.ipynb)
* [Naive Bayes and Gaussian Discriminant Analysis](https://github.com/puneet-08/projects/blob/main/Naive_Bayes_and_GDA_from_scratch.ipynb)
* [Support Vector Machines](https://github.com/puneet-08/projects/blob/main/SVM_from_scratch.ipynb)
* [K-Means Clustering](https://github.com/puneet-08/projects/blob/main/K_Means.ipynb)
* [Expectation Maximization](https://github.com/puneet-08/projects/blob/main/Expectation_Maximization.ipynb)


