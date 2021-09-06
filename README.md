# Pneumonia Detection

In this project, you can find a neural network trained on a dataset stored on [kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) to diagnose pneumonia based on a xray of the chest.

This project was part of the AI formation at [BeCode](https://becode.org/).

## The training dataset.

This dataset is isued from [this paper](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5). It consists of 5863 X-ray images (in jpeg format).
The images are labelled in two categories (Pneumonia/Normal). For Pneumonia images, there are also labels for the type of pneumonia (virus/bacteria) though this is not used in this project.
<p align="center">
    <img src="https://github.com/Nathanael-Mariaule/Pneumonia_Detection/blob/presentation/xray.jpg" width="150" height="150">
</p>

We used data augmentation to increase the size of the dataset and the performances of the model. We apply randoms zoom, shift, rotation and horizontal flips.

## The model

The model was build in Tensorflow using keras layers. It consists of 4 blocks of convolution and 1 block of dense layer. Each block of convolution is composed of 2 convolutions layers, a batchnormalization and a max-pooling. The dense blocks has one layer of 128 neurons and a single neuron ouput.
<p align="center">
    <img src="https://github.com/Nathanael-Mariaule/Pneumonia_Detection/blob/presentation/Model.png" width="100" height="300">
</p>

We trained the model for 15 epochs (with learning rate reductions).


## The results

On the test set, we get an accuracy of 90% with precision and recall of 87% and 98% resp. 

<p align="center">
    <img src="https://github.com/Nathanael-Mariaule/Pneumonia_Detection/blob/presentation/confusion_matrix.jpg">
</p>   
    
At this point, the model is not suitable for real-life production.Though we could improve our result with the following strategies:
-increase the dataset: a lot of medical data are available. We could use them to improve our dataset.
-transfer learning: we could use more complex models and obtain better result

