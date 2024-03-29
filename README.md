# NEURAL NETWORK EXPLORER
#### Video Demo:  [https://youtu.be/XOiG4eFewcU](https://youtu.be/XOiG4eFewcU)
#### Description:
Implement a neural network, and then explore several aspects of the network's operation.

# Overview
This project cotains two programs:
 1. **project.py**, which implements a neural network. The neural network "learns" how to recognize handwritten digits. It uses the MNIST (Modified National Institute of Standards and Technology) dataset. This program stores the neural network parameters in files in the **Neural-Network-Parameters** directory, using the JSON format. It also creates images for the Training Accuracy graph, Training Loss graph, Validation Accuracy graph, Validation Loss graph, and the Confusion Matrix.
 2. **Show-Neural-Network.py**, which allows you to explore the neural network created by the other program. It allows you to explore the neural network's connection weights and node biases. It lets you see the Training Accuracy graph, Training Loss graph, Validation Accuracy graph, Validation Loss graph, and the Confusion Matrix. Finally, it shows each instance where the Validation run got it's prediction wrong. 

# Neural Network
The neural network has the following characteristics:

| Characteristic | Details |
|----------------|---------|
| Layers: | 3 (Input Layer, Hidden Layer, and Output Layer) |
| Number of Input Nodes: | 784 |
| Number of Hidden Nodes: | 180 |
| Number of Output Nodes: | 10 |
| Activation Function: | Rectified Linear Unit (ReLU) |
| Loss Function: | Subtract one hot encoding of label from probabilities |
| Learning Rate: | 0.15 |

# Technologies
The technologies in this project are:
  - Python for server-side processing.
  - JSON to store the neural network's metadata, nodes, and connections information.
  - Flask to create the **Show-Neural-Network.py** web pages.
  - HTML and JavaScript to implement the web pages.
  - D3 to create the graphs showing the neural network's nodes and connections.
  - JQuery to dynamically load HTML (for the incorect predictions).
  - CSS to style the web pages.
  - GitHub for version control.

# Running the Programs
To run the **project.py** program, use the python command:
```
python Project.py
```

Before running the **Show-Neural-Network.py** program, you must ensure the ''FLASK_APP'' environment variable is set (in the Terminal window):
```
export FLASK_APP=Show-Neural-Network.py
```

To run the **Show-Neural-Network.py** program, use the flask command and then load http://127.0.0.1:5000/ in a browser:
```
flask run
```

# Using the Programs
There is no user interface for **project.py**. You can use the console to verify that the program runs correctly.

When you run **Show-Neural-Network.py** and load http://127.0.0.1:5000/ in a browser, you will see a web page with for tabs:
![Screenshot.png](Screenshot.png?raw=true)

Use these tabs to navigate between:
 - **Training Explorer**: Exploring the neural network's weights and biases.
 - **Validation Explorer**: Seeing the Confusion Matrix for the Validation run, as well as inforation aout each instance where the Validation run had an incorrect prediction.
 - **Accuracy & Loss:**: Seeing the Accuract graph for the Training and Validation runs, as well as the Loss graph for the Training and Validation runs.
 - **About the Neural Network**: Seeing information about the neural network hyperparameters.
 
 Here are some screenshots:

 ![Training-Explorer.png](Training-Explorer.png?raw=true)

 ![Validation-Explorer](Validation-Explorer.png?raw=true)

 ![Accuracy-and-Loss.png](Accuracy-and-Loss.png?raw=true)

# Files and Directories
This project has the following files and directories:

| File or Directory | Description |
|-------------------|-------------|
| train.csv | This CSV file contains the MNIST data set. That is, it includes the images of digits and the labels that correspond to those images. |
| project.py | This Python file implements and runs the neural network. For more deails about the neural network configuration, see the [Neural Network](#Neural-Network) section above. |
| Neural-Network-Parameters | This directory contains te JSON files that store the neural network's metadata, nodes, and connections information. | 
| requirements.txt | The pip-installable libraries this project requires. |
| Show-Neural-Network.py | This Python file creates the neural network explorer that allows you to see the connection weights, node biasesa, accuracy graphs, loss graphs, etc.|
| static/Accuracy.png | This image shows the accuracy graph for the Training and Validation runs. It is created by project.py. |
| static/Cofusion.png | This image shows the neural network's Confusion Matrix. It is created by project.py. |
| static/Loss.png | This image shows the loss graph for the Training and Validation runs. It is created by project.py. |
| static/BadPredictions.html | This file includes HTML with details about the incorrect predicions from the Validation run. It is dynamically loaded into project.py. |
| static/Instrumented-Neural-Networks.css | This is the CSS file for project.py. |
| Templates/index.html | This is the Flask template for the Show-Neural-Network.py home page. |
| Templates/layout.hml | This is the Flask template that lays out the Show-Neural-Network.py home page. |
| test_project.py | This is the unit test for Project.py. |



