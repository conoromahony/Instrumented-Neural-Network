# This code uses flask to create a web page that shows the operation of a neural network (that was created using 
# Simple-Neural-Network.py). It displays the entire neural network, while allowing you to zoom in on certain 
# neurons. It uses D3 to display the network graph.
# 
# Source 1: https://gist.github.com/e9t/6073cd95c2a515a9f0ba
# Source 2: https://codepen.io/Neo24/pen/GRRdBWr
# Looking at: https://d3-graph-gallery.com/graph/network_basic.html

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
#  - Switch it so we write to a TempFiles directory, and then ovwerwrite the contents of the directory on the next run.
#  - Is it possible to make this more efficient: read JSON --> create output_str --> create SVG

from flask import Flask, render_template, redirect, url_for, make_response
import numpy as np
import json


directory_name = "Neural-Network-Parameters-20240124-1941/"
file_name_base = "working-data-"
file_name_number = "0"
file_object = open(directory_name + file_name_base + file_name_number)
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
    # Metadata...
    output_str = "var newGraph = { 'metadata': [], 'nodes': [], 'connections':[] }; \n"
    for item in working_data["metadata"][0].keys():
        new_str = "var " + str(item) + " = '" + str(working_data["metadata"][0][item]) + "'; \n"
        output_str += new_str

    # Nodes:
    #  - Node ID (which is used for creating the links)
    #  - Layer #
    #  - Node #
    #  - Bias (db if backward step)
    for node in nodes_data:
        new_str = "var tempNode = { 'id': " + str(node["id"]) + ", 'layer': " + str(node["layer"]) + ", 'node': " + str(node["node"]) + ", 'bias': " + str(node["bias"]) + "}; newGraph.nodes.push(tempNode); \n"
        output_str += new_str

    # Connections:
    #  - Source node #
    #  - Target node #
    #  - Weight (dW if backward step)
    for link in connections_data:
        new_str = "var tempLink = { 'source': " + str(link["source"]) + ", 'target': " + str(link["target"]) + ", 'weight': " + str(link["weight"]) + "}; newGraph.connections.push(tempLink); \n"
        output_str += new_str

    return render_template("index.html", network_graph=output_str)

@app.route('/next')
def next_iteration():
    # TODO: Is it a problem that I am not closing the file, before opening another?
    # The following line tells this module to use the global value of file_name_number in this function.
    global file_name_number
    # TODO: Need to check haven't gone past last iteration!!!
    file_name_number = str(int(file_name_number) + 1)
    file_object = open(directory_name + file_name_base + file_name_number)
    working_data = json.load(file_object)
    meta_data = working_data["metadata"]
    nodes_data = working_data["nodes"]
    connections_data = working_data["connections"]
    iteration_number = meta_data[0]["iteration_number"]
    direction = meta_data[0]["direction"]
    output_str = "var newGraph = { 'metadata': [], 'nodes': [], 'connections':[] }; \n"
    for item in working_data["metadata"][0].keys():
        new_str = "var " + str(item) + " = '" + str(working_data["metadata"][0][item]) + "'; \n"
        output_str += new_str
    for node in nodes_data:
        new_str = "var tempNode = { 'id': " + str(node["id"]) + ", 'layer': " + str(node["layer"]) + ", 'node': " + str(node["node"]) + ", 'bias': " + str(node["bias"]) + "}; newGraph.nodes.push(tempNode); \n"
        output_str += new_str
    for link in connections_data:
        new_str = "var tempLink = { 'source': " + str(link["source"]) + ", 'target': " + str(link["target"]) + ", 'weight': " + str(link["weight"]) + "}; newGraph.connections.push(tempLink); \n"
        output_str += new_str
    return render_template("index.html", network_graph=output_str)


@app.route('/previous')
def previous_iteration():
    # TODO: Is it a problem that I am not closing the file, before opening another?
    # The following line tells this module to use the global value of file_name_number in this function.
    global file_name_number
    # TODO: Need to check haven't gone before first iteration!!!
    file_name_number = str(int(file_name_number) - 1)
    file_object = open(directory_name + file_name_base + file_name_number)
    working_data = json.load(file_object)
    meta_data = working_data["metadata"]
    nodes_data = working_data["nodes"]
    connections_data = working_data["connections"]
    iteration_number = meta_data[0]["iteration_number"]
    direction = meta_data[0]["direction"]
    output_str = "var newGraph = { 'metadata': [], 'nodes': [], 'connections':[] }; \n"
    for item in working_data["metadata"][0].keys():
        new_str = "var " + str(item) + " = '" + str(working_data["metadata"][0][item]) + "'; \n"
        output_str += new_str
    for node in nodes_data:
        new_str = "var tempNode = { 'id': " + str(node["id"]) + ", 'layer': " + str(node["layer"]) + ", 'node': " + str(node["node"]) + ", 'bias': " + str(node["bias"]) + "}; newGraph.nodes.push(tempNode); \n"
        output_str += new_str
    for link in connections_data:
        new_str = "var tempLink = { 'source': " + str(link["source"]) + ", 'target': " + str(link["target"]) + ", 'weight': " + str(link["weight"]) + "}; newGraph.connections.push(tempLink); \n"
        output_str += new_str
    return render_template("index.html", network_graph=output_str)


# *** CURRENTLY: adding buttons to step through training
# *** Add checks for iterating too far eithger way
# *** Add button to go to start or end
# *** Add button to play / pause through iterations
# *** Maintain node close-up view when going to next or previous

# *** Maybe generate mathplotlib images upon training runs, store them in the folder, and display them in the UI?
# ***  Training error rate
# ***  Validation error rate
# ***  Test error rate