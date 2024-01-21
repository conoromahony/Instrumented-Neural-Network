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
#  - git commit -m "message"
#  - git push

# TODO:
#  - Combine this file with the file that creates and runs the neural network (i.e. Simple-Neural-Network.py). 

from flask import Flask, render_template, make_response
import numpy as np

directory_name = "Neural-Network-Parameters-20240121-1326/"
file_name = "working-data-0"

nodes_in_layer_0 = 784
nodes_in_layer_1 = 64
nodes_in_layer_2 = 10

weights = np.load(directory_name + file_name)

app = Flask(__name__)

@app.route("/")
def hello():
    output_str = "var newGraph = { 'nodes': [] }; \nvar newInputLayer = []; \nvar newHiddenLayer = []; var newOutputLayer = []; \n"
    for i in range(0, nodes_in_layer_0):
        new_str = "var newTempLayer = { 'label': " + str(i) + ", 'layer': 0 }; newInputLayer.push(newTempLayer); \n"
        output_str += new_str
    for j in range(0, nodes_in_layer_1):
        new_str = "var newTempLayer = { 'label': " + str(j) + ", 'layer': 1 }; newHiddenLayer.push(newTempLayer); \n"
        output_str += new_str
    for k in range(0, nodes_in_layer_2):
        new_str = "var newTempLayer = { 'label': " + str(k) + ", 'layer': 2 }; newOutputLayer.push(newTempLayer); \n"
        output_str += new_str
    output_str += "newGraph.nodes = newGraph.nodes.concat(newInputLayer, newHiddenLayer, newOutputLayer);"
    return render_template("index.html", network_graph=output_str)


# Just getting started pulling in serialized data.
# Of course, will need to unserialize it.
# Will also need to wor with the new format.