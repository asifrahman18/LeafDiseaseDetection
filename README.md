<h1>Motivation</h1>

The motivation of this project is totally based on the target user of
gardening and the severity of the leaf disease. It is quite clear that
gardening has become a common practice for urban life. Nowadays,
especially in cities we can hardly find a roof that has no plant in it.
But still, we cannot neglect the fact that most of the urban people
involved in gardening has no prior knowledge about farming. They do not
know what are the diseases the plants can face, how to detect these
diseases and finally how to remedy them. Hence, they often fail to
detect the disease in time. Also, leaf disease can be both infectious
and non-infectious depending on the nature of a causative agent. If not
detected early it can lead to the death of the plant which could have
easily been solved by a touch of technology. But most of the features of
agricultural technology focuses mainly on field agricultural crops
rather that residential gardening. The reason why we are taking this
project is to help these urban people get acquainted with a
technological tool that will help them to detect the plant diseases in
real time.

**3.1 Workflow**

In our project, our model is a CNN model. A workflow diagram displaying
the complete process of the method is shown below:

<!---
![](vertopal_c705d2cb40af4be3a9ce76b28f9316fb/media/image1.png){width="3.563888888888889in"
height="4.093055555555556in"}
-->

*Figure 1: Block diagram of workflow*

In the diagram, we can see that our first step is collecting a suitable
dataset which we can use for training and testing our model. The images
from the dataset have to be pre-processed before we can use it for
training and testing. After training the model, we can test the model by
making predictions on some images and validating the prediction with the
actual image and also find the accuracy.

**3.2 Data Collection**

Collecting dataset is our first step and also one of the most important
step. Before collecting a dataset, there are some features that has to
be explored. Since our project requires an image dataset, we have to
look at features like the number of image samples, variations of image,
background, image format and resolution.

We have collected our dataset from **\[8\]** which contains .jpg type of
image. There are ten classes of five different types of plants
containing 2273 pictures of both healthy and diseased leaves. All the
images in the dataset has similar background. The resolution of all
images are also same.

**3.3 Pre-processing**

Before using our images for training, the images have to be
pre-processed. Pre-processing is required to increase the quality of
input image so that the performance of the model can be improved. We
followed the following steps for pre-processing our images:

<!---
![](vertopal_c705d2cb40af4be3a9ce76b28f9316fb/media/image2.png){width="2.8652777777777776in"
height="3.645832239720035in"}

*Figure 2: Block diagram of image pre-processing*
-->
From the above diagram, we can see that after obtaining the image from
the dataset, we will be resizing the images. Then we will normalize the
images and finally apply image augmentation.

**3.3.1 Resizing**

We started our pre-processing with image rescaling and resizing. Our
images from the dataset were 6000x6000 resolution. We resized the image
into 256x256 size to ensure that all images stay at a consistent scale.

**3.3.2 Normalization**

After resizing the images, we normalized the images. Normalization keeps
all the input pixels on similar distribution. In our model, we performed
normalization by dividing the pixel values by 255. By dividing the pixel
values, we rescale the values between 0 and 1.

For our model, we performed normalization implicitly in the resizing and
rescaling step.

**3.3.3 Image Augmentation**

Our final step of pre-processing stage is image augmentation. Data
augmentation is applied to the existing dataset to artificially increase
the size of the dataset by performing various transformations like
flipping, distorting, rotating, zooming etc. Data augmentation helps to
create variation and lowers the chances of overfitting.

In our model, we have randomly flipped the images horizontally and
vertically. We have also rotated the images randomly up to 0.2 radians.

**3.4 Building The Model**

Our Machine Learning model is a CNN model. Our model can be divided into
two parts, Feature extraction and image classification. We have used CNN
for both feature extraction and image classification.

**3.4.1 Feature Extraction**

Feature extraction transforms raw data into numeric values compatible
with machine learning\
algorithms **\[9\]**. These numeric values can be processed by keeping
the integrity of the information that is found from the original
dataset. For our project, we used CNN for feature extraction. There are
two main layers of feature extraction in a CNN model. If we take a look
at the CNN architecture,

<!---
![](vertopal_c705d2cb40af4be3a9ce76b28f9316fb/media/image3.png){width="4.708333333333333in"
height="2.2597222222222224in"}

*Figure 3: CNN model architecture of feature extraction*

-->
Here we can see that there are two layers in feature extraction,
Convolutional layer and Pooling layer.

Convolutional layer helps to detect various features and pattern which
are present in the image by applying convolutional filters to the input
image. This filters then learn extracting different visual features like
textures.

Pooling layer selects the minimum value within a specific region of the
image and down samples it. This helps to retain the relevant information
of the image by reducing the relative dimension.

By stacking these two layers, the model can learn how to identify
features from the input image.

Our model also has these two layers. We have used two convolution layers
and two pooling layers. The convolution layers of our model have 32
filters and each filter has 3x3 kernel size. We have used the activation
filter ReLU (Rectified Linear Unit) which is a popular activation
function. ReLU uses the formula,

f(x) = max(0,x)

Which means that the output can be found only if the input value is
positive, otherwise it will be zero. If there is any negative input,
ReLU deactivates the neuron by converting it to zero\[\]. As a result,
exponential growth in computation can be prevented. Our two pooling
layers capture the most important features with a pool size of 2x2.

**3.4.2 Image Classification**

In a CNN model, the classification layer determines the predicted value
on the activation map **\[10\]**. Once the model has extracted features,
it needs to classify those features into learn the features of the data.

CNN classification also has two layers. If we look at the classification
architecture,

<!---
![](vertopal_c705d2cb40af4be3a9ce76b28f9316fb/media/image4.png){width="4.688888888888889in"
height="2.3430555555555554in"}

*Figure 4: CNN model architecture of classification*

-->
We can see that there is a flatten layer and a dense layer.

The flatten layer flattens the 2D layers from feature extraction into 1D
vector. This layer converts the features into a format which can be
processed by the dense layers.

After flattening, all the layers are passed to the dense layer or fully
connected layer. Dense layer contains multiple neurons which are fully
connected to each other. This layers learns to classify the extracted
features into their corresponding classes.

For image classification, we have also used these two layers. The
flatten layer flattens the 2D feature maps into a 1D vector before going
to the dense layer. Our first dense layer is a fully connected layer
consisting of 64 units or neurons. This layer has ReLU activation
function which applies linear\
transformation to the input data. Our second dense layer is the output
layer of the model. We have used softmax activation function here. Each
output value of this layer represents the probability of input of a
particular class.

<!---
![](vertopal_c705d2cb40af4be3a9ce76b28f9316fb/media/image5.png){width="3.7180555555555554in"
height="3.6375in"}

*Figure 5: Model summary*

-->
**3.5 Training The Model**

After preprocessing and building the model, our model is ready for
training. We have split our dataset into two partitions. The first
partition is our training dataset. Which is 80% of the total dataset.
And the remaining 20% is the testing dataset. We further split the
testing dataset into two more partitions. Which are test and validation.
Each partition is 10% of the testing dataset.

We started training our model with the train dataset. The validation
dataset is used to evaluate the model's performance. We used the loss
function Sparse Categorical Crossentropy. This function calculates the
cross-entropy loss between the true labels and the predicted labels. The
value increase when the difference increases. We also used Adam
optimizing algorithm. Adam can adapt the learning rate of each parameter
also maintaining a moving average of squared gradients. As a result,
Adam provides stable and fast optimization.

At first, we started training our model with a batch size of 32. We also
had an epoch value of 50, which is the number of iterations the model
will go through over the entire training dataset. With this values, our
did not have a stable accuracy. So, we had to tune the parameters. We
tried different batch sizes, epoch values, and learning rate of the
optimizer and found the best accuracy with a batch size of 32, epoch 30
and learning rate of 0.001.

**3.6 Testing The Model and Making Predictions**

After completing the training part, we tested our model and made
predictions. At first, we evaluated the model on the test dataset and
found the loss and accuracy values. For prediction, at first we selected
one image sample from test dataset and used the model to make
predictions on it. The model was able to make correct predictions. Then
we selected 9 images from the test dataset and made predictions. This
time, all the predictions were mostly correct. We also displayed the
confidence level by taking the maximum predict probability.

**Results**

Confusion matrix is one of the most common performance measurement used
in machine learning. It shows the summary of a model's performance on a
test data. This matrix uses the true labels and the predicted labels to
show four types of information. These are True Positive (TP), True
Negative (TN), False Positive (FP) and False Negative (FN)

<!---
![](vertopal_c705d2cb40af4be3a9ce76b28f9316fb/media/image6.png){width="6.097222222222222in"
height="5.583333333333333in"}

*Figure 6: Confusion matrix*
-->

From the confusion matrix, we can also calculate some important data for
each class which we can use to analyze the model. Which are Accuracy,
Specificity, Sensitivity, Precision and F1 score.

Accuracy shows the outputs that are actually correct. It is calculated
using the formula,

> 𝑇𝑃 + 𝑇𝑁\
> 𝑇𝑃 + 𝐹𝑁 + 𝐹𝑃 + 𝑇𝑁

Specificity is the model's performance on identifying true negative
cases, which is calculated using the

following formula,

> 𝑇𝑁\
> 𝑇𝑁 + 𝐹𝑃

Sensitivity is the model's performance on identifying true positive
cases. Sensitivity is also known as

recall. This is calculated using the formula,

> 𝑇𝑃\
> 𝑇𝑃 + 𝐹𝑁

Precision shows how many true positives are predicted correctly. Which
means finds how many true

positives are actually true positives. The formula which is used to
calculate precision is,

> 𝑇𝑃\
> 𝑇𝑃 + 𝐹𝑃

F1 score can be calculated from precision and Recall. It is the harmonic
mean of the precision and recall

**\[11\]**, which is calculated by,

> 2𝑇𝑃\
> 2𝑇𝑃 + 𝐹𝑃 + 𝐹𝑁

We have calculated all these values for each class of our model from the
Confusion Matrix, and got the following scores,

+-----------+-----------+-----------+-----------+-----------+-----------+
| > **Class | > **A     | > **Spec  | > **Sens  | > **Pr    | > **F1    |
| > Name**  | ccuracy** | ificity** | itivity** | ecision** | > Score** |
+===========+===========+===========+===========+===========+===========+
| > guava   | > 0.973   | > 0.981   | > 0.846   | > 0.733   | > 0.785   |
| _diseased |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > guav    | > 0.822   | > 1.00    | > 0.870   | > 1.00    | > 0.931   |
| a_healthy |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > jamun   | > 0.822   | > 0.983   | > 0.973   | > 0.925   | > 0.948   |
| _diseased |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > jamu    | > 0.733   | > 0.990   | > 0.809   | > 0.894   | > 0.850   |
| n_healthy |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > lemon   | > 0.995   | > 0.995   | > 1.00    | > 0.800   | > 0.888   |
| _diseased |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > lemo    | > 0.995   | > 1.00    | > 0.909   | > 1.00    | > 0.952   |
| n_healthy |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > mango   | > 0.991   | > 0.989   | > 1.00    | > 0.941   | > 0.969   |
| _diseased |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > mang    | > 1.00    | > 1.00    | > 1.00    | > 1.00    | > 1.00    |
| o_healthy |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| po        | > 0.982   | > 0.980   | > 1.00    | > 0.851   | > 0.920   |
| megranate |           |           |           |           |           |
| _diseased |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| p         | > 0.982   | > 1.00    | > 0.862   | > 1.00    | > 0.925   |
| omegranat |           |           |           |           |           |
| e_healthy |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+

*Table 1: Model evaluation matrices*

We also have two graphs which can show the performance of our model. The
first Graph is the Training and Validation accuracy graph. This graph
shows the progression of accuracy of the model over each epoch. The
training accuracy is the accuracy of the model on the training dataset
and the validation accuracy is the accuracy on the validation dataset
which is different from the training dataset.

<!---
![](vertopal_c705d2cb40af4be3a9ce76b28f9316fb/media/image7.png){width="6.5in"
height="2.1944444444444446in"}

*Figure 7: Training and validation accuracy graph*


From the graph, we can see that the accuracy starts to increase as the
model learns to adjust the parameters. We can also see that the training
and validation curves are increasing at similar level which shows that
the model is learning well.

Our other graph is Training and Validation loss graph. This graph shows
the change loss of the model over each epoch. The goal here is to
minimize the amount of loss over each epoch.


![](vertopal_c705d2cb40af4be3a9ce76b28f9316fb/media/image8.png){width="6.5in"
height="2.1763877952755903in"}

*Figure 8: Training and validation loss graph*

In this graph, we can see that the loss amount starts to decrease with
each epoch. The training and validation curves are also on similar
position.

-->

\[1\] "City heat can be reduced by Rooftop Gardens", Prothomalo.
\[Accessed: April 8, 2023\]\
Available at:\
\[2\] "Leaf spot" (2022) Wikipedia. Wikimedia Foundation \[Accessed:
April 8, 2023\] Available at:\
\[3\] Horst, R. (2008). Westcott's Plant Disease Handbook (Seventh
Edition.). Springer Netherlands. \[4\] Ethem Alpaydin (2020).
Introduction to Machine Learning (Fourth Ed.). MIT. pp. xix, 1--3,
13--18. ISBN 978-0262043793.

\[5\] Leaf Disease Detection Using Convolutional Neural Network, Sneha
Patel\
\
\[Published: July 2022\]\
\[6\] Singh, V., & Misra, A. (2017). Detection of plant leaf diseases
using image segmentation and soft computing techniques. *Information
Processing in Agriculture*, *4*(1), 41-49.

\[7\] Kulkarni, Pranesh & Karwande, Atharva & Kolhe, Tejas & Kamble,
Soham & Joshi, Akshay & Wyawahare, Medha. (2021). Plant Disease
Detection Using Image Processing and Machine Learning.\
\[8\] Healthy vs. Diseased Leaf Image Dataset\
\
\[9\] The Role of Feature Extraction in Machine Learning \| Snowflake.
Snowflake.

\[10\] *What Is a Convolutional Neural Network? A Beginner\'s Tutorial
for Machine Learning and Deep Learning*. FreeCodeCamp.

\[11\] Wikipedia contributors. (2023, April 4). F-score. In *Wikipedia,
The Free Encyclopedia*. Retrieved 00:45, June 16, 2023, from
