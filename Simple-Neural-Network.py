# Implements a simple two-layer neural network. Input layer 𝑎[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image. 
# A hidden layer 𝑎[1] will have 64 units with ReLU activation, and finally our output layer 𝑎[2] will have 10 units corresponding to the ten digit
# classes with softmax activation.
# Video: https://www.youtube.com/watch?v=w8yWXqWQYmU
# Blog post: https://www.samsonzhang.com/2020/11/24/understanding-the-math-behind-neural-networks-by-building-one-from-scratch-no-tf-keras-just-numpy

# This code uses Gradient Descent. The basic idea of gradient descent is to figure out what direction each parameter can go in to decrease error
# by the greatest amount, then nudge each parameter in its corresponding direction over and over again until the parameters for minimum error and
# highest accuracy are found. In a neural network, gradient descent is carried out via a process called backward propagation. We take a prediction,
# calculate an error of how off it was from the actual value, then run this error backwards through the network to find out how much each weight
# and bias parameter contributed to this error. Once we have these error derivative terms, we can nudge our weights and biases accordingly to improve
# our model. Do it enough times, and we'll have a neural network that can recognize handwritten digits accurately.

# To run:
#  - Go to Desktop > Programming > Instrumented-Neural-Network
#  - Type "python Simple-Neural-Network.py"
# 
# To commit changes:
#  - Edit with Visual Studio
#  - git add *
#  - git commit -m "message"
#  - git push

# TODO:
#  - See if there's a way to add X, Z1, A1, Z2, and A2 to the serialized working data.
#  - Add the back propagation data to the serialized working data.
#  - Refactor the code so the number of layers is not hardcoded.
#  - See if it makes sense to serialize the matrices directly, and shift the processing of them to the Javascript code
#  - Add other activation functions.
#  - Consider variations of gradient descent that improve training efficiency: gradient descent with momentum, RMSProp, and Adam optimization. 
#  - Create a package for the refactored neural network code.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker
import os, shutil
import json


num_input_nodes = 784
num_hidden_layers = 1
num_hidden_nodes = 64
num_output_nodes = 10
num_iterations = 30
activation_fn = "Rectified Linear Unit (ReLU)"
alpha_value = 0.2
loss_fn = "Subtract a one hot encoding of the label from the probabilities"

training_accuracy = []
validation_accuracy = []


# We will place our files in the "Neural-Network-Parameters" directory. If the directory does not exist, create it.
# If the directory exists, clear its contents. In the directory, we will have one JSON file for each iteration (epoch). 
# We will also store images for the test and validation error rates.
directory_name = "Neural-Network-Parameters"
if not os.path.isdir(directory_name):
    os.makedirs(directory_name)
for filename in os.listdir(directory_name):
    file_path = os.path.join(directory_name, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))    


# Create the initial weights and biases for the neural network.
# Note: it's best practice to initialize your weights/biases close to 0, otherwise your gradients get really small really quickly:
# https://stackoverflow.com/questions/47240308/differences-between-numpy-random-rand-vs-numpy-random-randn-in-python
def init_params():
    # Defines the weights for the conections to the nodes in layer 1. W1 is a 64 x 784 matrix with random values.
    # We subtract 0.5 from the random values so we end up with numbers between -0.5 and 0.5 (rather than 0 and 1),
    W1 = np.random.rand(num_hidden_nodes, num_input_nodes) - 0.5
    # Defines the biases for the nodes in layer 1. b1 is a 64 x 1 matrix with random values.
    b1 = np.random.rand(num_hidden_nodes, 1) - 0.5
    # Defines the weights for the conections to the nodes in layer 2. W2 is a 10 x 64 matrix with random values.
    W2 = np.random.rand(num_output_nodes, num_hidden_nodes) - 0.5
    # Defines the biases for the nodes in layer 2. W1 is a 10 x 1 matrix with random values.
    b2 = np.random.rand(num_output_nodes, 1) - 0.5
    return W1, b1, W2, b2


# Implement the Rectified Linear Unit (ReLU) function. That is, a simple linear function that returns:
#   x if x > 0
#   0 if x <= 0
def ReLU(Z):
    return np.maximum(Z, 0)


# Implement the softmax function. That is, it translates the values to probabilities, between 0 and 1, that all add up to 1.
# Softmax takes a column of data at a time, taking each element in the column and outputting the exponential of that element divided by the
# sum of the exponentials of each of the elements in the input column.
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


# Perform forward propagation for the hidden and ouput layers.
#   𝑍[1] = 𝑊[1]𝑋+𝑏[1]
#   𝐴[1] = 𝑔ReLU(𝑍[1]))
#   𝑍[2] = 𝑊[2]𝐴[1]+𝑏[2]
#   𝐴[2] = 𝑔softmax(𝑍[2])
def forward_prop(W1, b1, W2, b2, X):
    # Calculate the node values for layer 1 (the hiden layer). Remember W1 is a numpy array, so we can use .dot for matrix operations.
    # W1 is a 64x784 matrix. X is a 784x41000 matrix. Their dot product is a 64x41000 matrix. Therefore, Z1 is a 64x41000 matrix.
    Z1 = W1.dot(X) + b1
    # Apply the activation function. We are using the Rectified Linear Unit (ReLU) function.
    A1 = ReLU(Z1)
    # Calculate the node values for layer 2 (the output layer).
    # W2 is a 10x64 matrix. A1 is a 64x41000 matrix. Their dot product is a 10x41000 matrix. Therefore, Z2 is a 10x41000 matrix.
    Z2 = W2.dot(A1) + b2
    # Apply the softmax function. The softmax function turns the output values into probabilities.
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# Implement the derivative of the activation function (i. the ReLU function).
# Note that the slope of the ReLU function when X is less than zero is 0, and the slope of the ReLU function when X is greater than zero is 1.
def ReLU_deriv(Z):
    # When booleans convert to numbers, true converts to 1 and false converts to 0.
    # Z > 0 is true when any one element of Z is greater than 0 (ie. the function returns 1)
    # Z > 0 is false when no element of Z is greater than 0 (i.e. the function returns 0)
    return Z > 0


# Implement "one hot" encoding for the labels in the training data. That is, create a matrix for all images, where each column represents an image label.
# Put 1 in the position of the label, and 0's in all other positions.
def one_hot(Y):
    # Create an m x 10 matrix.  Y.size is m.  Y.max() is 9 (i.e. the biggest value when working with the digits 0-9 is 9).
    # Initialize the matrix to have zeros in all positions.
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # For each row identified by np.arange(Y.size), change the value in column Y to 1.
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Transpose the matrix, so each column represents an image label. That is, return a 10 x m matrix.
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


# Perform back propagation through the neural network. 
# Here are the calcuations for the weights and biases for layer 2 (i.e. the output layer)
#   𝑑𝑍[2]=𝐴[2]−𝑌      To determine the error for the output layer during training (i.e. dZ2), subtract a "one hot encoding" of the label from the probabilities.
#   𝑑𝑊[2]=1/𝑚 . 𝑑𝑍[2]𝐴[1]𝑇      That is, the average of the error values.
#   𝑑𝐵[2]=1/𝑚 . Σ𝑑𝑍[2]        
# Here are the calcuations for the weights and biases for layer 1 (i.e. the hidden layer)
#   𝑑𝑍[1]=𝑊[2]𝑇 . 𝑑𝑍[2].∗𝑔[1]′(𝑧[1])     Taking error from layer 2 (i.e. dZ2), and applying weights to it in reverse (i.e. transpose of W2). g' is the drivative of the activation function.
#   𝑑𝑊[1]=1/𝑚 . 𝑑𝑍[1]𝐴[0]𝑇
#   𝑑𝐵[1]=1/𝑚 . Σ𝑑𝑍[1]
# Note that one commenter wrote that... I believe dZ[2] should be 2(A[2]−Y) because the error/cost at the final output layer should be (A[2]−Y)^2. 
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    # The closer the prediction probability is to 1, the closer the loss is to 0. By minimizing the cost function, we improve the accuracy of our model.
    # We do so by substracting the derivative of the loss function with respect to each parameter from that parameter over many rounds of graident descent.
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


# Update our parameters as follows:
#   𝑊[2]:=𝑊[2]−𝛼𝑑𝑊[2]
#   𝑏[2]:=𝑏[2]−𝛼𝑑𝑏[2]
#   𝑊[1]:=𝑊[1]−𝛼𝑑𝑊[1]
#   𝑏[1]:=𝑏[1]−𝛼𝑑𝑏[1]
# Alpha is the learning rate. Alpha is a hyper parameter (i.e. it is not trained by the model).
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


# Return the first column of A2.
def get_predictions(A2):
    return np.argmax(A2, 0)


# Get the accuracy between the predictions (i.e. A2) and Y (i.e. the labels).
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# This pulls everything together. It initializes the parameters, performs the forward propagation, the backward propagation, and updates the parameters.
# It does this iteration times, and it prints out an update every 10 iterations.
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):        
        # Create the data structures for storing the details of the neural network working data.
        # For each iteration, we will have one working data file. The name of the file will indicate the iteration.
        # The working_data dictionary will have three lists: meta_data, node_data, and connections_data.
        #
        # Metadata:
        #  - Number of hidden layers
        #  - Number of nodes in each layer
        #  - Number of iterations
        #  - Iteration
        #  - Direction (i.e. forward or backward)
        #  - Activation function (i.e. descriptive text)
        #  - Alpha
        #  - Prediction
        #  - Label (i.e. the actual value)
        #  - Loss function (i.e. descriptive text)
        #
        # Neurons:
        #  - Layer #
        #  - Node #
        #  - ID # (which is used for creating the links)
        #  - Bias (db if backward step)
        # 
        # Connections:
        #  - Source neuron node #
        #  - Target neuron node #
        #  - Weight (dW if backward step)
        #
        # Note: I don't see a practcal way to include the X (training data), Z1, A1, Z2, or A2 values. During each iteration, we process
        # 41,000 images. That means X is a 784x41000 matrix. In other words, during each iteration, we process 41,000 values through each
        # node in the network. This also means 41,000 values of Z1, A1, Z2, and A2 for each iteration. I'm not sure how to gracefully show
        # this. For now, I will not include this information in the JSON. Maybe she can show this informatn for the "inference" phase, rather
        # than the training phase.

        working_data = {}
        meta_data = []
        nodes_data = []
        connections_data = []

        # Creating the data structure that stores the meta data for the working data
        temp_metadata = {}
        temp_metadata["num_input_nodes"] = num_input_nodes
        temp_metadata["num_hidden_layers"] = num_hidden_layers
        temp_metadata["num_hidden_nodes"] = num_hidden_nodes
        temp_metadata["num_output_nodes"] = num_output_nodes
        temp_metadata["num_iterations"] = num_iterations
        temp_metadata["iteration_number"] = i
        temp_metadata["direction"] = "forward"
        temp_metadata["activation_fn"] = activation_fn
        temp_metadata["alpha_value"] = alpha_value
        temp_metadata["prediction"] = ""
        temp_metadata["actual_value"] = ""
        temp_metadata["loss_fn"] = loss_fn
        meta_data.append(temp_metadata)

        # Creating the data structure that stores the working data for the connections between nodes in the input layer and the hidden layer
        for temp_i in range(1, num_input_nodes):
            for temp_j in range(1, num_hidden_nodes):
                temp_connection = {}
                temp_connection["source"] = 10000 + temp_i       # To make the node IDs unique I am adding a number indicating the layer
                temp_connection["target"] = 20000 + temp_j       # To make the node IDs unique I am adding a number indicating the layer
                temp_connection["weight"] = W1[temp_j,temp_i]
                connections_data.append(temp_connection)
        # Creating the data structure that stores the working data for the connections between nodes in the hidden layer and the output layer
        for temp_k in range(1, num_hidden_nodes):
            for temp_l in range(1, num_output_nodes):
                temp_connection = {}
                temp_connection["source"] = 20000 + temp_k       # To make the node IDs unique I am adding a number indicating the layer
                temp_connection["target"] = 30000 + temp_l       # To make the node IDs unique I am adding a number indicating the layer
                temp_connection["weight"] = W2[temp_l,temp_k]
                connections_data.append(temp_connection)

        for temp_m in range(1, num_input_nodes):
            temp_node = {}
            temp_node["layer"] = 0
            temp_node["node"] = temp_m
            temp_node["id"] = 10000 + temp_m
            temp_node["bias"] = 0                               # There is no bias for the input nodes
            nodes_data.append(temp_node)
        for temp_n in range(1, num_hidden_nodes):
            temp_node = {}
            temp_node["layer"] = 1
            temp_node["node"] = temp_n
            temp_node["id"] = 20000 + temp_n
            temp_node["bias"] = b1[temp_n, 0]
            nodes_data.append(temp_node)
        for temp_o in range(1, num_output_nodes):
            temp_node = {}
            temp_node["layer"] = 2
            temp_node["node"] = temp_o
            temp_node["id"] = 30000 + temp_o
            temp_node["bias"] = b2[temp_o, 0]
            nodes_data.append(temp_node)

        working_data["metadata"] = meta_data
        working_data["nodes"] = nodes_data
        working_data["connections"] = connections_data

        # Serializing the JSON data
        json_object = json.dumps(working_data, indent=4)
 
        # Writing JSON data to file
        file_name =  directory_name + "/working-data-" + str(i)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

        # Store the Training Accuracy data for this epoch, so we can graph it later.
        # A2 are the predictions that come out the other end of forward propagation.
        # Y are the image labels.
        predictions = get_predictions(A2)
        training_accuracy.append(get_accuracy(predictions, Y))
        # Store the Validation Accuracy data for this epoch, so we can graph it later.
        dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
        validation_accuracy.append(get_accuracy(dev_predictions, Y_dev))

        # TODO: Add the back propagation data to the serialized working data.

        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(predictions, Y))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


# Test the neural network's prediction for the image at the "index" parameter.
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# We are using the MNIST digit recognizer dataset. MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world”
# dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking
# classification algorithms. Use pandas to read the CSV file with the data.
data = pd.read_csv('train.csv')
# Use numpy to load the CSV data into an array.
data = np.array(data)
# Get the dimensions of the array. There are m rows (i.e. images). Each image has n (i.e. 785 values; one for the label and 784 for the pixels)
m, n = data.shape

# Shuffle the data before splitting into dev and training sets.
np.random.shuffle(data)

# Create the dev data (i.e. validation data) from the first 1,000 images.
# Remember to transpose the matrix, so each column (rather than row) is an image.
data_dev = data[0:1000].T
# Now, Y_dev (i.e. the image label) will just be the first row.
Y_dev = data_dev[0]
# And X_dev will be the image pixels.
X_dev = data_dev[1:n]
# The pixel valueas (0-255) are transformed into decimal values (0-1).
X_dev = X_dev / 255.

# Create the training data from the remaining images. There are something like 41,000 of them.
# Again, remember to transpose the matrix so each column is an image.
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
# The pixel valueas (0-255) are transformed into decimal values (0-1).
X_train = X_train / 255.

# Run the neural network on the training set.
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha_value, num_iterations)

# Test the neural network's prediction for the images at indexes 0, 1, 2, and 3.
#test_prediction(0, W1, b1, W2, b2)
#test_prediction(1, W1, b1, W2, b2)
#test_prediction(2, W1, b1, W2, b2)
#test_prediction(3, W1, b1, W2, b2)

iteration_array = np.arange(0, num_iterations)
training_array = np.array(training_accuracy)
validation_array = np.array(validation_accuracy)
plt.plot(iteration_array, training_array, color='r', label="Training Accuracy")
plt.plot(iteration_array, validation_array, color='g', label="Validation Accuracy")
plt.title("Training - Accuracy at each Epoch", fontweight='bold')
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy")
plt.legend(loc='best', frameon=False)
formatter = matplotlib.ticker.PercentFormatter(xmax=1)
plt.gca().yaxis.set_major_formatter(formatter)
if os.path.isdir("static/Accuracy.png"):
    os.remove("static/Accuracy.png")
plt.savefig("static/Accuracy.png")

# Need to write the JSON data.
# Following instructions in: https://www.geeksforgeeks.org/reading-and-writing-json-to-a-file-in-python/
# Taking inspiration from: https://d3-graph-gallery.com/network.html
# Sample JSON file: https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/data_network.json
# I think I have the writing of the JSON data working, although there are some TODOs above.
# Time to now focus on creating the network diagram via JavaScript.