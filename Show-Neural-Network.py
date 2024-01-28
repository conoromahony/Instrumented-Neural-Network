# This code uses flask to create a web page that shows the operation of a neural network (that was created using 
# Simple-Neural-Network.py). It displays the entire neural network, while allowing you to zoom in on certain 
# neurons. It uses D3 to display the network graph.
# 
# Source 1: https://gist.github.com/e9t/6073cd95c2a515a9f0ba
# Source 2: https://codepen.io/Neo24/pen/GRRdBWr

# To run:
#  - Go to Desktop > Programming > Instrumented-Neural-Network
#  - Type "export FLASK_APP=Show-Neural-Network.py"
#  - Type "flask run"
#  - To view, load the following page in a browser: http://127.0.0.1:5000/
# 
# To commit changes:
#  - Edit with Visual Studio
#  - git add *
#  - git commit -m "message"
#  - git push

# TODO:
#  - Combine this file with the file that creates and runs the neural network (i.e. Simple-Neural-Network.py). 

from flask import Flask, render_template, make_response
import numpy as np
import json


directory_name = "Neural-Network-Parameters-20240124-1941/"
file_name = "working-data-0"
file_object = open(directory_name + file_name)
working_data = json.load(file_object)
meta_data = working_data["metadata"]
nodes_data = working_data["nodes"]
connections_data = working_data["connections"]


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
num_input_nodes = meta_data[0]["num_input_nodes"]
num_hidden_layers = meta_data[0]["num_hidden_layers"]
num_hidden_nodes = meta_data[0]["num_hidden_nodes"]
num_output_nodes = meta_data[0]["num_output_nodes"]
num_iterations = meta_data[0]["num_iterations"]
iteration_number = meta_data[0]["iteration_number"]
direction = meta_data[0]["direction"]
activation_fn = meta_data[0]["activation_fn"]
alpha_value = meta_data[0]["alpha_value"]
prediction = meta_data[0]["prediction"]
actual_value = meta_data[0]["actual_value"]
loss_fn = meta_data[0]["loss_fn"]


app = Flask(__name__)

@app.route("/")
def hello():
    # Neurons:
    #  - Layer #
    #  - Node #
    #  - ID # (which is used for creating the links)
    #  - Bias (db if backward step)
    output_str = "var newGraph = { 'nodes': [], 'connections':[] }; \n"
    for node in nodes_data:
        new_str = "var tempNode = { 'id': " + str(node["id"]) + ", 'layer': " + str(node["layer"]) + ", 'node': " + str(node["node"]) + ", 'bias': " + str(node["bias"]) + "}; newGraph.nodes.push(tempNode); \n"
        output_str += new_str
    # Connections:
    #  - Source neuron node #
    #  - Target neuron node #
    #  - Weight (dW if backward step)
    for link in connections_data:
        new_str = "var tempLink = { 'source': " + str(link["source"]) + ", 'target': " + str(link["target"]) + ", 'weight': " + str(link["weight"]) + "}; newGraph.connections.push(tempLink); \n"
        output_str += new_str
    return render_template("index.html", network_graph=output_str)


# I could create a Neural Network Health Check of sorts:
#  - If a weight is close to zero, it essentially represents a connection between two nodes that is rarely, if ever, used.
#    These connections are rarely, if ever, used. They don't contribute to the networ because the signals don't pass. If a 
#    network has a lot of such weights, perhaps it is using a sub-optimal configuration. This is something worth exploring
#    and potentially reporting.

# Just getting started pulling in serialized data.
# Have pulled in the metadata.
# Now need to get started on the nodes and connections.
# Current approach is to create JavaScript in this file to render the network graph, and pass that JavaScript to the HTML file.
