# This code uses flask to create a web page that shows the operation of a neural network. It displays the entire neural 
# network, while allowing you to zoom in on certain neurons. It uses D3 to display the network graph.

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
#  - Switch it so we write to a TempFiles directory, and then ovwerwrite the contents of the directory on the next run.
#  - Is it possible to make this more efficient: read JSON --> create output_str --> create SVG

from flask import Flask, render_template, flash
import numpy as np
import json


directory_name = "Neural-Network-Parameters/"
file_name_base = "working-data-"
file_name_number = "0"
file_object = open(directory_name + file_name_base + file_name_number)
working_data = json.load(file_object)
meta_data = working_data["metadata"]
nodes_data = working_data["nodes"]
connections_data = working_data["connections"]


num_input_nodes = meta_data[0]["num_input_nodes"]           # Number of nodes in input layer
num_hidden_layers = meta_data[0]["num_hidden_layers"]       # Number of hidden layers
num_hidden_nodes = meta_data[0]["num_hidden_nodes"]         # Number of nodes in hidden layer
num_output_nodes = meta_data[0]["num_output_nodes"]         # Number of nodes in output layer
num_iterations = meta_data[0]["num_iterations"]             # Number of iterations (epochs)
iteration_number = meta_data[0]["iteration_number"]         # Current iteration (epoch))
direction = meta_data[0]["direction"]                       # Direction (i.e. forward or backward)
activation_fn = meta_data[0]["activation_fn"]               # Activation function (i.e. descriptive text)
alpha_value = meta_data[0]["alpha_value"]                   # Alpha
prediction = meta_data[0]["prediction"]                     # Prediction
actual_value = meta_data[0]["actual_value"]                 # Label (i.e. the actual value)
loss_fn = meta_data[0]["loss_fn"]                           # Loss function (i.e. descriptive text)


app = Flask(__name__)

@app.route("/")
def hello():
    # This route generates the code for the home page.
    # Generate the code that stores the metadata...
    output_str = "var newGraph = { 'metadata': [], 'nodes': [], 'connections':[] }; \n"
    for item in working_data["metadata"][0].keys():
        new_str = "var " + str(item) + " = '" + str(working_data["metadata"][0][item]) + "'; \n"
        output_str += new_str
    # Generate the code that specifies the node data. For each node, it includes:
    #  - Node ID (which is used for creating the links)
    #  - Layer #
    #  - Node #
    #  - Bias (db if backward step)
    for node in nodes_data:
        new_str = "var tempNode = { 'id': " + str(node["id"]) + ", 'layer': " + str(node["layer"]) + ", 'node': " + str(node["node"]) + ", 'bias': " + str(node["bias"]) + "}; newGraph.nodes.push(tempNode); \n"
        output_str += new_str
    # Generate the code that specifies the connections data. For each connection, it includes:
    #  - Source node #
    #  - Target node #
    #  - Weight (dW if backward step)
    for link in connections_data:
        new_str = "var tempLink = { 'source': " + str(link["source"]) + ", 'target': " + str(link["target"]) + ", 'weight': " + str(link["weight"]) + "}; newGraph.connections.push(tempLink); \n"
        output_str += new_str
    return render_template("index.html", network_graph=output_str)

@app.route('/first')
def first_iteration():
    # This route generates the code when the "First Epoch" button is clicked.
    # TODO: Is it a problem that I am not closing the file, before opening another?
    # The following lines tells this module to use the global values of variables in this function.
    global file_name_number
    global num_iterations
    file_name_number = "0"
    file_object = open(directory_name + file_name_base + file_name_number)
    working_data = json.load(file_object)
    nodes_data = working_data["nodes"]
    connections_data = working_data["connections"]
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
    # This route generates the code when the "Previous Epoch" button is clicked.
    # TODO: Is it a problem that I am not closing the file, before opening another?
    # The following line tells this module to use the global value of file_name_number in this function.
    global file_name_number
    if int(file_name_number) > 0:
        file_name_number = str(int(file_name_number) - 1)
    else:
        file_name_number = "0"
    file_object = open(directory_name + file_name_base + file_name_number)
    working_data = json.load(file_object)
    nodes_data = working_data["nodes"]
    connections_data = working_data["connections"]
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

@app.route('/next')
def next_iteration():
    # This route generates the code when the "First Epoch" button is clicked.
    # TODO: Is it a problem that I am not closing the file, before opening another?
    # The following lines tells this module to use the global values of variables in this function.
    global file_name_number
    global num_iterations
    if int(file_name_number) < int(num_iterations) - 1:
        file_name_number = str(int(file_name_number) + 1)
    else:
        file_name_number = str(int(num_iterations) - 1)
    file_object = open(directory_name + file_name_base + file_name_number)
    working_data = json.load(file_object)
    nodes_data = working_data["nodes"]
    connections_data = working_data["connections"]
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

@app.route('/last')
def last_iteration():
    # This route generates the code when the "Last Epoch" button is clicked.
    # TODO: Is it a problem that I am not closing the file, before opening another?
    # The following lines tells this module to use the global values of variables in this function.
    global file_name_number
    global num_iterations
    file_name_number = str(num_iterations - 1)
    file_object = open(directory_name + file_name_base + file_name_number)
    working_data = json.load(file_object)
    nodes_data = working_data["nodes"]
    connections_data = working_data["connections"]
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


# *** Maybe generate mathplotlib images upon training runs, store them in the folder, and display them in the UI?
# ***  Training error rate
# ***  Validation error rate
# ***  Test error rate

# *** Why are there only 9 nodes in the output layer, 63 nodes in the hidden layer?