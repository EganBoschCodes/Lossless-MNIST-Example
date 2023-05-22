package main

import (
	"fmt"
	"os"
	"time"

	"github.com/EganBoschCodes/lossless/datasets"
	"github.com/EganBoschCodes/lossless/neuralnetworks/layers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/networks"
)

func prepareData() {
	testFrame := datasets.ReadCSV("mnist_test.csv", true)
	testFrame.NumericallyCategorizeColumn("label")
	testFrame.MapFloatColumnSlice("[1:]", func(_ int, a float64) float64 { return a / 255 })
	testData := testFrame.ToDataset("[1:]", "[0]")

	trainingFrame := datasets.ReadCSV("mnist_train.csv", true)
	trainingFrame.NumericallyCategorizeColumn("label")
	trainingFrame.MapFloatColumnSlice("[1:]", func(_ int, a float64) float64 { return a / 255 })
	trainingData := trainingFrame.ToDataset("[1:]", "[0]")

	datasets.SaveDataset(testData, "data", "mnist_test")
	datasets.SaveDataset(trainingData, "data", "mnist_training")

	fmt.Println("\nTest and Training data saved!")
}

func train() {
	trainingData, testData := datasets.OpenDataset("data", "mnist_training"), datasets.OpenDataset("data", "mnist_test")

	network := networks.Sequential{}
	network.Initialize(784,
		&layers.Conv2DLayer{
			InputShape:  layers.Shape{Rows: 28, Cols: 28},
			KernelShape: layers.Shape{Rows: 3, Cols: 3},
			NumKernels:  6,
			FirstLayer:  true,
		},
		&layers.MaxPool2DLayer{
			PoolShape: layers.Shape{Rows: 2, Cols: 2},
		},
		&layers.TanhLayer{GradientScale: 2.0},
		&layers.Conv2DLayer{
			InputShape:  layers.Shape{Rows: 13, Cols: 13},
			KernelShape: layers.Shape{Rows: 3, Cols: 3},
			NumKernels:  12,
		},
		&layers.TanhLayer{GradientScale: 2.0},
		&layers.FlattenLayer{},
		&layers.LinearLayer{Outputs: 100},
		&layers.TanhLayer{GradientScale: 2.0},
		&layers.LinearLayer{Outputs: 10},
		&layers.SoftmaxLayer{},
	)

	network.BatchSize = 32
	network.LearningRate = 0.02

	network.Train(trainingData, testData, time.Second*60)

	network.Save("savednetworks", "MNIST_Network")

}

func main() {
	switch len(os.Args) {
	case 1:
		train()
	case 2:
		if os.Args[1] == "-prep" || os.Args[1] == "-p" {
			prepareData()
		} else {
			panic(os.Args[1] + " is not a valid flag (only -prep works)")
		}
	default:
		panic("this file only takes 0 or 1 arguments!")
	}
}

func toASCII(values []float64) string {
	colors := []string{"  ", "░░", "▒▒", "▓▓", "██"}
	stringVal := ""

	for i, val := range values {
		if i%28 == 0 && i != 0 {
			stringVal = fmt.Sprint(stringVal, "\n")
		}
		index := int(val * 4.99)
		stringVal = fmt.Sprint(stringVal, colors[index])
	}

	return stringVal
}

func PrintLetter(letter datasets.DataPoint) {
	fmt.Println("Printing Digit:", datasets.FromOneHot(letter.Output))
	fmt.Println(toASCII(letter.Input))
	fmt.Printf("Output: %.2f\n", letter.Output)
}
