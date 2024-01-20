# To run:
#  - Go to Desktop > Programming > Instrumented-Neural-Network
#  - Type "flask run"
# To commit changes:
#  - Edit with Visual Studio
#  - git commit -m "message"
#  - git push

from flask import Flask, render_template, make_response
import numpy as np

directory_name = "Neural-Network-Parameters-20240117-1441/"
file_name = "W1-0.npy"

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

# Using flask. To run, simply enter "python Interpret.py" and then view http://127.0.0.1:5000/
# Creating a webpage that has the weights between the nodes on Layer 0 and the nodes on Layer 1
# That's it. No other weights. No biases. 
# And I'm only doing it for the first iteration (i.e. before we have even trained any data).
# Next up... do a better job of displaying this data (i.e. these weights).
# Find some visual way to present it.
# Maybe have two areas of screen: the left side shows entire structure; the right side shows close up of one node.
# Then let's figure out how to do it for all of the iterations. (Perhaps via a drop-down list of iterations.)

# Source 1: https://gist.github.com/e9t/6073cd95c2a515a9f0ba
# Source 2: https://codepen.io/Neo24/pen/GRRdBWr
    

# Make sure to include D3:
#   <script src="https://d3js.org/d3.v7.min.js"></script>
# D3 supports different types of data like arrays, CSV, XML, TSV, JSON, and so on.
# This data can come from a local file in your working directory