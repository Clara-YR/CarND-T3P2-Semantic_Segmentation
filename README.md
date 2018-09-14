
[image0]: ./readme_images/save&load_model.png "how to load model"
[image1]: ./readme_images/vgg_model.png "vgg 16 model structure"
[image2]: ./readme_images/vgg_layers.png "vgg 16 model structure"
[image3]: ./readme_images/Segmentation_Architecture.png "vgg 16 model structure"
[image4]: ./readme_images/run.png "python main.py output"
[image5]: ./readme_images/regularizer.png "no regularizer v.s. regularizer"

# Semantic Segmentation

### Setup

**My package version**

- python 3.6.4
- TensorFlow 1.10.1
- NumPy 1.14.0
- SciPy 1.0.0

Tips: Use `pip install --upgrade ...(==special version)` to upgrade TensorFlow, Numpy and Scipy to the latest version.

**Download DataSet**

Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/data_road.zip). Extract the dataset in the data folder. This will create the folder data_road with all the training and 
test images.

<p style='color:red'>This is some red text.</p>
<font color="red">This is some text!</font>

These are <b style='color:red'>red words</b>.


# <font color="green">1. Build the Neural Network</font>

## <font color="green">1.1 Load and Pretrained VGG Model</font>
_This part depicts how the project loads the pretrained vgg model._


### <font color="green">1.1.1 VGG Model</font>

Reference download from [link](https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/):

![alt text][image1]
![alt text][image2]

Then we get infromation as below:

|Layer|Output Shape|
|:---:|:----------:|
|layer __3__|(None, 28, 28, __256__)|
|layer __4__|(None, 14, 14, __512__)|
|layer __7__|(None, __4096__)|

Additional material: [transfer learning](https://github.com/udacity/deep-learning/blob/master/transfer-learning/Transfer_Learning_Solution.ipynb)

### <font color="green">1.1.2 Function `load_vgg`</font>

![alt text][image0]


## <font color="green">1.2 Learn Image Features</font>

_This part depicts how the project learns the correct features from the images._

### <font color="green"> 1.2.1 Segmentation Architecturein</font>

![alt text][image3]

**Pseudocode**

- 1x1 Convolution
- Deconvolution
	- up-sample by 2(use <b style='color:green'>regularizer</b>)
	- combine current and previous pool4 outputs
	- up-sample by 2 again(use <b style='color:green'>regularizer</b>)
	- combine current and previous pool3 outputs
	- up-sample by 8(use <b style='color:green'>regularizer</b>)
- return layers output	

**Regularizer**

The upper and uder images displays the training output with and without using regularizer respectively.
![alt text][image5]
If we don't use regularizer:
 
 - weights will become too large
 - the NN is prone to overfitting and producing garbage

**Padding**

Set all paddings as `same` since we want teh same exact size as the output.


### <font color="green"> 1.2.2 Implement FCN-8 in `layers`</font>

__Reference links__:

- [`tf.layers.conv2d()`](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)
- [`tf.layers.conv2d_transpose()`](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d_transpose)
- [`tf.layers.dense()`](https://www.tensorflow.org/api_docs/python/tf/layers/dense)
- [`tf.add()`](https://www.tensorflow.org/api_docs/python/tf/add)

|    |Encoder|1x1 conv|Decoder|
|:--:|:-----:|:------:|:-----:|
|structure|pre-trained vgg|use `l2_regularizer`|transpose convolution|
|function|extract features|preserve special indormation|up-sample|

#### <font color="green">(1) FCN - Encoder</font>

The encoder for FCN-8 is the VGG16 model pretrained on ImageNet for classification. 

The fully-connected layers 

```
output = tf.layers.dense(input, num_classes)
```
are replaced by 1-by-1 convolutions.

```
output = tf.layers.conv2d(input, num_classes, 1, strides=(1,1))
```

#### <font color="green">(2) FCN - Decoder</font>

Output of the final convolutional transpose layer will be 4-dimensional: (batch_size, original_height, original_width, num_classes).

```
output = tf.layers.conv2d_transpose(input, num_classes, 4, strides=(2, 2))
```


##### <font color="green">__Skip Connections__</font>

- Retaining the information easilly
	- multiple resolutions
	- more precise segmentation decisions
- How: connect the output of one layer to a non-adjacent layer
- output of encoder `pool` __*__ current layers output
	- __*__ element-wise addition operation
	
```
# make sure the shapes are the same!
input = tf.add(input, pool_4)
input = tf.layers.conv2d_transpose(input, num_classes, 4, strides=(2, 2))

input = tf.add(input, pool_3)
Input = tf.layers.conv2d_transpose(input, num_classes, 16, strides=(8, 8))
```


## <font color="green">1.3 Optimization</font>
_This part depicts how the project optimizes the neural network._

### <font color="green">Function `optimize`</font>
__Reference Links__:

- [`tf.nn.softmax_cross_entropy_with_logits()`](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)
- [`tf.train.AdamOptimizer()`](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)


#### <font color="green">(1) FCN - Classification</font>
```
logits = tf.reshape(input, (-1, num_classes))
```

#### <font color="green">(2) FCN - Loss</font>
```
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
```

#### <font color="green">(3) Train Optimizer</font>
```
optimizer = tf.train.AdamOptimimzer(learning_rate)
train_op = optimizer.minimize(cost)
```


## <font color="green">1.4 Train the Neural Network</font>
_The loss of the network should be printed while the network is training._

### <font color="green">1.4.1 Function `train_nn`</font>

I add a parameter `progressbar` in order to display the progress of each epoch when training.

**Pseudocode**

- Initialize
- For each epoch
	- Get batches, for each batch
		- Feed dictionay
		- Run optimizer


# <font color="purple">2. Train the Neural Network</font>
_This part depicts how sthe project trains the neural network._

## <font color="purple">2.1 Hyperparameters</font>
_The number of epoch and batch size are set to a reasonable number._

I set the parameters as below:

- __learning rate = 0.1__  (line 173)
- __keep probability = 0.8__  (line 172)
- __epochs = 6__  (line 232)
- __batch size = 10__  (line 233)


## <font color="purple">2.2 Function 'run'</font>

**Pseudocode**

- Implement the functions created above in order:
	- `load_vgg`
	- `layers`
	- `optimize`
	- `train_nn`
- Save the training output via `helper.save_inference_samples`

My output shown as below:
![alt text][image4]




Other Syntax Redeference:

- [`tqdm`](https://pypi.python.org/pypi/tqdm) - A fast, extensible progress bar for Python and CLI 
- [`urllib.urlretrieve(url[, filename[, reporthook[, data]]])`](https://docs.python.org/2/library/urllib.html?highlight=urlretrieve#urllib.urlretrieve)
- [`yield`](https://docs.python.org/2.4/ref/yield.html)