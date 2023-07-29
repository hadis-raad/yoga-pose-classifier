# Yoga Pose Classifier using Transfer Learning

The objective of this project is to detect five different Yoga poses: downdog, goddess, plank, tree, and warrior2. The dataset used for this classification task is the <a href="https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset">Yoga Poses Dataset</a> from Kaggle.

One of the challenges faced in this project is the limited number of images per class in the training data, with only around 200 images per pose. To address this issue and improve classification accuracy, a Transfer Learning approach using the ResNet50V2 model is adopted. Transfer Learning allows us to leverage the knowledge gained from pre-trained models on large datasets and apply it to our task, enabling more efficient and accurate classification.


The data have been uniformly rescaled to [224, 224] to ensure that all images in the dataset have the same size.
<p align="center"> 
 <smaller> <i>Figure 1. Yoga poses</i></smaller>
</p>
<p align="center">
  <img src="https://github.com/hadis-raad/yoga-pose-classifier/blob/main/images/Yoga_poses.png" alt="axis of movement" width="450" />
</p>

An additional complexity arises from the fact that some variations of the five poses can closely resemble each other. Poses and some variations are shown in Figure 1.

In this regard, data augmentation techniques are employed to address this challenge and improve model accuracy. After training the model and visualizing the confusion matrix and ROC curve, incorrect predictions are identified. To further enhance the model's performance, additional data augmentation techniques are implemented.

<p align="center"> 
 <smaller> <i>Figure 2. Data Augmentation</i></smaller>
</p>
<p align="center">
  <img src="https://github.com/hadis-raad/yoga-pose-classifier/blob/main/images/data_augmentation.png" alt="data augmented images" width="400"/>
</p>

The dataset used for training, validation, and testing consists of:

* Training data: 974 images
* Validation data: 107 images
* Test data: 470 images

To improve the ResNet50V2 model's performance, additional layers were added to the architecture in addition to data augmentation techniques.

```
x = Conv2D(128, (3, 3), activation='relu')(resent_model.output)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(128,activation='relu')(x)
x = Dense(5,activation='softmax')(x)
```


Moreover, to prevent overfitting during training, the ReduceLROnPlateau method was utilized to dynamically adjust the learning rate based on the validation loss.


```
lrr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001)
```

By fine-tuning hyperparameters and optimizing the model, the final version successfully achieved <b>90% accuracy</b> on the test data.

The resulting confusion matrix and ROC curve are depicted below, illustrating the model's ability to correctly classify the different Yoga poses, except goddess pose. The model misclassified 28 instances of warrior2 pose. By analyzing the data, we identified that some data had been mislabeled. Therefore, one may achieve better accuracy by reevaluating the labeling of some data.

<p align="center"> 
 <smaller> <i>Figure 3. Confusion matrix and ROC curve of Yoga poses classifier model </i></smaller>
</p>


<img src="https://github.com/hadis-raad/yoga-pose-classifier/blob/main/images/confusion_matrix.png" alt="confusion matrix" width="400"/><img src="https://github.com/hadis-raad/yoga-pose-classifier/blob/main/images/roc.png" alt="roc curve" width="450"/>





