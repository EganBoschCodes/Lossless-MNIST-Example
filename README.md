

# CNNs in Lossless - MNIST Dataset
Now, everyone loves a good old classic perceptron, but machine learning (and Lossless) has a lot more to offer. Convolutional neural networks, or CNNs, do very well at image recognition, by essentially filtering for different characteristics in an input image. 

A classic dataset to get introduced to CNNs is the MNIST dataset, a dataset containing 60,000 training and 10,000 testing datapoints representing handwritten digits from 0 - 9. Each input is a 28x28 image, constituting 784 total inputs, and each pixel has a single value from 0 - 255. In this example project, we will train a convolutional neural network on the MNIST dataset, going over every line of code from
## Prerequisites
1. Download the latest version of Go [here](https://go.dev/dl/).
2. Create your project directory, and run `git clone https://github.com/EganBoschCodes/Lossless-MNIST-Example`
3. Run `go get github.com/EganBoschCodes/lossless` to install Lossless.
4. Download the two MNIST .csv's from Kaggle [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv), and place into the top project directory.
## Data Preparation
### Reading in the Data
Before we can train our network, we need to import our data and properly format it. The `datasets.DataFrame` class is a great tool for this, and you can see how we use it in the `prepareData` method in `main.go`. In order to create our dataframe, we call:
```Go
func  prepareData() {
	trainingFrame  := datasets.ReadCSV("mnist_training.csv", true)
	...
}
```
This takes the CSV at the given address and populates the dataframe, and passing the second argument `true` lets it know the first row is just the headers for the columns. Feel free to run `trainingFrame.PrintSummary()` immediately after we import the data to see what it looks like. It should look like:
```bash
 label |  1x1  |  1x2  |  1x3  |  1x4  |  1x5  |  1x6  |  1x7  |  1x8  |  1x9   (773 more columns...)
-------------------------------------------------------------------------------
 7.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 
 2.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 
 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 
 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 
 4.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 
 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 
 4.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 
 9.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 
 5.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 
 9.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 

                                (59990 more rows...)
```
Just note, we see a whole lot of zeros in the columns after the first because these represent the values just in top row, left corner. If you want to see some other values, try running:
```Go
subframe  := trainingFrame.SelectColumns("[0, 295:302]")
subframe.PrintSummary()
```
This will select just the first column, and then the 295th-301st columns and print those. You'll see numbers ranging from 0 - 255 now. You'll notice that the syntax for column selection is very similar to how array slicing works in Go, and it is: you can just add commas to do multiple slices in one call. This allows for easy manipulation of the data.

### Data Normalization

For example, let's normalize our values to be between 0 and 1, instead of 0 and 255, as you don't want inputs of too large a magnitude into your neural networks generally. This just takes one line of code:
```Go
trainingFrame.MapFloatColumnSlice("[1:]", func(_ int, a float64) float64 { return a /  255 })
```
This will iterate over ever column from the one in the 1st index (I.E. the second column) all the way out to the end, just like with normal Go slicing syntax. Then, for ever value in every one of the selected columns, it will apply our given lambda function, which just divides all values by 255. You can check that this worked by reselecting and printing the subframe again after this line of code; you should now only see values between 0 and 1. Perfect!

You can also call `means, stddevs := trainingFrame.NormalizeColumnSlice("[1:]")` if you'd instead wish to normalize each column based off their means and standard deviations. The variables `means` and `stddevs` now contain a list of all the means and standard deviations used when normalizing your data, so that if you wish to map future inputs to the same distribution as your training set you can use those. However, in our case we will remain for faithful to the original idea of an image, and so not normalize each column.

### Creating One-Hot Vectors

Now we need to take that label column and turn it into a vector output instead of just a number, because that's the type of output our neural network needs. We can run:
```Go
trainingFrame.NumericallyCategorizeColumn("label")
```
This will iterate over the `label` column, find the maximum value, then replace every value in the label column with a one hot vector with the same length as the maximum value representative of the value that was there. For example, `7` becomes `[0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]`, with zeroes everywhere but a one in the 7th index. Feel free to check this with another `trainingFrame.PrintSummary()`.

If instead of having numbers in our label column we had strings, like `"one"`, `"two"`, etc. we could do:
```Go
mappings := trainingFrame.CategorizeColumn("label")
```
This will output the list of strings corresponding to each index in the one-hot vector. So, for example, you could get back `mappings = ["apple" "orange" "banana"]`, which means that if your network pops out `[0 1 0]`, it thinks it's an orange.

However, since in our case we already had nice numbers which can intuitively map to indexes, we just go with numerical categorization.
### Turning DataFrames into Trainable Datasets
Now that our DataFrame is good and processed, we can go ahead and just turn it into a dataset nice and easily.
```Go
trainingData  := trainingFrame.ToDataset("[1:]", "[0]")
```
This specifies that our input will be the columns from the 1st index all the way to the final column, and that our output is just the first column. Just like that, we have our dataset!

Now that we've properly formatted our data, you can try out the pre-prepared `PrintLetter` function by calling:
```Go
PrintLetter(trainingData[0])
```
You'll get a nice visualization of the kind of stuff we are working with!

### Saving Processed Datasets
Doing all this data processing can take a few moments; even on my very good machine, everything takes like 5 seconds in total. We don't want to wait for that to happen every time we boot up to do some more training, so Lossless lets you save your datasets to a file to avoid that situation. Simply call:
```Go
datasets.SaveDataset(trainingData, "data", "mnist_training")
```
This will create a folder titled `data` in your project directory, and save your dataset into a `mnist_training.dtst` file for later use.
### Running this Example Code
In order to run this code we just went over, type:
```shell
go run main.go -p
```
The `-p` is not a flag that Lossless understands and makes you only prepare your data, but if you look at `main()` you'll just see that the `-p` flag makes you call the `prepareData()` function instead of the `train()` one. I would recommend separating your code in a similar fashion to this as well.

## Training a CNN
Now it's time for the big moment! We are going to train a convolutional neural network on the data we just processed and prepared for it. Let's get started!
### Loading back in our Saved Data
Luckily, this is also very simple. Just call the function:
```Go
trainingData := datasets.OpenDataset("data", "mnist_training")
testData := datasets.OpenDataset("data", "mnist_test")
```
It will check the data folder for your datasets with the given names and load them into memory nice and fast.
### Building a CNN
Now this requires a little bit of an understanding of how convolutional neural networks function. An excellent resource that I would recommend reading before continuing with this tutorial can be found [here](https://saturncloud.io/blog/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way/).

So now hopefully you are familiar with the idea of convolving a kernel across an input image, as that is exactly what we are about to do. So let's get started! First, we must create our network.
```Go
network  := networks.Sequential{}
```
Then we must initialize it with our call to the `Initialize` method. We know that our input is a 28 by 28 image, so that means our network will take in 28 x 28 = 784 inputs in total.
```Go
network.Initialize(784,
	...
)
```
Now, we begin with a `Conv2DLayer`, which essentially is a standard convolution for an image of a single channel. Since our input image only contains greyscale input, this will do just fine. We need to decide on the structure of this layer though. We already know that it will take a 28x28 image as input, and lets go with a 3x3 kernel and 6 kernels. We can also let this layer know that it is the first layer, letting it know it doesn't have to bother calculating backpropagation gradients to pass back another layer.
```Go
network.Initialize(784,
	&layers.Conv2DLayer{
		InputShape: layers.Shape{Rows: 28, Cols: 28},
		KernelShape: layers.Shape{Rows: 3, Cols: 3},
		NumKernels: 6,
		FirstLayer: true,
	},
	...
)
```
Now we will pass our output from that layer, which will be 6 26x26 images, through a `MaxPool2DLayer`, making sure we only keep analyzing the most important features of the output and reducing the amount of computational work necessary down the line. For this pool, we will just pool in a 2x2 sample.
```Go
network.Initialize(784,
	...
	&layers.MaxPool2DLayer{
		PoolShape: layers.Shape{Rows: 2, Cols: 2},
	},
	...
)
```
Now we have 6 13x13 images ready to keep being processed. We'll go for another round convolutions, this time with 12 kernels, so that each of our 6 images get two unique kernels. 

But first, before we keep passing it on, we need to pass it through an activation function. Many CNNs use ReLu activations, but empirically I have found that Tanh activations seem to do better, so we will use a Tanh layer. We will pass it an optional parameter, `GradientScale`, which essentially multiplies the gradients that the layer passes back by the given constant, effectively allowing you to have a larger learning rate later in the network with reference to earlier in the network. This option defaults to 1, which produces the expected gradients, but again I have empirically found that having a `GradientScale` greater than one helps combat the vanishing gradient problem and allows my networks to learn faster.
```Go
network.Initialize(784,
	...
	&layers.TanhLayer{GradientScale: 2.0},
	&layers.Conv2DLayer{
		InputShape: layers.Shape{Rows: 13, Cols: 13},
		KernelShape: layers.Shape{Rows: 3, Cols: 3},
		NumKernels: 12,
	},
	...
)
```
Now after this second convolution, we have 12 11x11 images. Lets activate them again, and then we will flatten our input. This will turn our 12 11x11 images into a single long 1452 vector that can be passed to linear layers.
```Go
network.Initialize(784,
	...
	&layers.TanhLayer{GradientScale: 2.0},
	&layers.FlattenLayer{},
	...
)
```
Finally, we finish up with some fully connected linear layers in a manner that is similar to a classic perceptron.
```Go
network.Initialize(784,
	...
	&layers.LinearLayer{Outputs: 100},
	&layers.TanhLayer{GradientScale: 2.0},
	&layers.LinearLayer{Outputs: 10},
	&layers.SoftmaxLayer{},
)
```
Since our final linear layer outputs 10 values, we have an output matching what we're expecting, which means it's training time! As always, feel free to tweak the hyperparameters to whatever works best for you; I generally go with a `BatchSize` of 32 and a `LearningRate` of 0.02. All that's left is to call our training method!
```Go
network.Train(trainingData, testData, time.Second*60)
```
Feel free to train for longer, as this is a larger network and it can benefit from it, but for now you should see something along the lines of:
```bash
Beginning Loss: 4500.249
Correct Guesses: 860/10000 (8.60%)

Training Progress : -{▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒}- (100.0%)  
Final Loss: 113.608
Correct Guesses: 9853/10000 (98.53%)
```
This means our network, after training on our training data for just a minute, is able to accurately classify 98.53% of the testing data! As an exercise, feel free to call the `network.GetErrors(testData)` method and use the print letter method to see what letters our network classifies incorrectly. Some of them a human could maybe get, but some would have even stumped me.
### Saving your Network
Of course we don't want to have to sit through another minute of training every time we want to use our nice and trained network, so we can save it with:
```Go
network.Save("path/where/you/want/it", "MNIST_Network")
```
And whenever we decide we want to use it in the future, we can just call:
```Go
network.Open("path/where/it/was", "MNIST_Network")
```
## Get Creating!
Now that you have this reference, feel free to continue to explore the power of CNNs on some other datasets!