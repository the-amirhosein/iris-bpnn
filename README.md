# iris-bpnn

In the implementation, the available data were used to classify irises. Iris flowers are divided
into physical characteristics. Each flower of this species has different appearance characteristics.
versicolor and virginica, setosa are three categories in the available feature database measured in 
four cases. Seedle length, sepal width, petal length and petal width.

Given that the number of available features is 4, so the number of cells in the input layer will be 4
cells, which with a bias cell we will have 5 cells. In the hidden layer, according to the tests performed
with different number of cells, the network with 7 cells in the hidden layer will have the best result
. In the number of layers tests, 5 to 9 cells were tested, which is the best result for 7 cells. 
In the output layer, due to the existence of 3 species, we will have three groups of neurons.
The active function used in this implementation is the Sigmoid function