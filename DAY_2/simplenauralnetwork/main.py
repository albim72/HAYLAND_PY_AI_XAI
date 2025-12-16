import  numpy as np
from simplenn import SimpleNeuralNetwork

network = SimpleNeuralNetwork()
print(network)

#train dataset
train_inputs = np.array([[1,1,0],[1,1,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0],[0,0,0]])
train_outputs = np.array([[1,0,1,1,0,1,1]]).T
train_iterators = 50_000

#trianing
network.train(train_inputs, train_outputs, train_iterators)
print(f"\nweights after training: \n{network.weights}")

#test data
testdata = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])

#prediction
print("prediction with traied model\n")
for data in testdata:
    print(f"result for {data} -> {network.propagation(data)}")

