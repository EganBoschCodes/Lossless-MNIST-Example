# CNNs in Lossless - MNIST Dataset
Now, everyone loves a good old classic perceptron, but machine learning (and Lossless) has a lot more to offer. Convolutional neural networks, or CNNs, do very well at image recognition, by essentially filtering for different characteristics in an input image. 

A classic dataset to get introduced to CNNs is the MNIST dataset, a dataset containing 60,000 training and 10,000 testing datapoints representing handwritten digits from 0 - 9. Each input is a 28x28 image, constituting 784 total inputs, and each pixel has a single value from 0 - 255. In this example project, we will train a convolutional neural network on the MNIST dataset, going over every line of code from
## Prerequisites
1. Download the latest version of Go [here](https://go.dev/dl/).
2. Create your project directory, and run `git clone https://github.com/EganBoschCodes/Lossless-MNIST-Example`
3. Run `go get github.com/EganBoschCodes/lossless` to install Lossless.
4. Download the two MNIST .csv's from Kaggle [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv), and place into the top project directory.
## Data Preparation
Before we can train our network, we need to import our data and properly format it. The `datasets.DataFrame` class is a great tool for this, and you can see how we use it in the `prepareData` method in `main.go`. In order to create our dataframe, we call:
```Go
func  prepareData() {
	testFrame  := datasets.ReadCSV("mnist_test.csv", true)
	...
}
```
This takes the CSV at the given address and populates the dataframe, and passing the second argument `true` lets it know the first row is just the headers for the columns. Feel free to run `testFrame.PrintSummary()` immediately after we import the data to see what it looks like. It should look like:
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

                                 (9990 more rows...)
```
Just note, we see a whole lot of zeros in the columns after the first because these represent the values just in top row, left corner. If you want to see some other values, try running: