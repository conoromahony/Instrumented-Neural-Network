# NEURAL NETWORK EXPLORER
#### Video Demo:  <URL HERE>
#### Description:


# Overview
This project cotains two programs:
 1. **Simple-Neural-Network.py**, which implements a neural network. The neural network "learns" how to recognize handwritten digits. It uses the MNIST (Modified National Institute of Standards and Technology) dataset. This program stores the neural network parameters in files in the **Neural-Network-Parameters** directory, using the JSON format. It also creates images for the Training Accuracy graph, Training Loss graph, Validation Accuracy graph, Validation Loss graph, and the Confusion Matrix.
 2. **Show-Neural-Network.py**, which allows you to explore the neural network created by the other program. It allows you to explore the neural network's connection weights and node biases. It lets you see the Training Accuracy graph, Training Loss graph, Validation Accuracy graph, Validation Loss graph, and the Confusion Matrix. Finally, it shows each instance where the Vaidation run got it's prediction wrong. 

# Neural Network
The neural network has the following characteristics:

| Characteristic | Details |
|----------------|---------|
| Layers: | 3 (Input Layer, Hidden Layer, and Output Layer) |
| Number of Input Nodes: | 784 |
| Number of Hidden Nodes: | 180 |
| Number of Output Nodes: | 10 |
| Activation Function: | Rectified Linear U nit (ReLU) |
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
To run the **Simple-Neural-Network.py** program, use the python command:
```
python Simple-Neural-Network.py
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
There is no user interface for **Simple-Neural-Network.py**. You can use the console to verify that the program runs correctly.

When you run **Show-Neural-Network.py** and load http://127.0.0.1:5000/ in a browser, you will see a web page with for tabs:
![Screenshot.png](Screenshot.png?raw=true)

Use these tabs to navigate between:
 - **Training Explorer**: Exploring the neural network's weights and biases.
 - **Validation Explorer**: Seeing the Confusion Matrix for the Validation run, as well as inforation aout each instance where the Validation run had an incorrect prediction.
 - **Accuracy & Loss:**: Seeing the Accuract graph for the Training and Validation runs, as well as the Loss graph for the Training and Validation runs.
 - **About the Neural Network**: Seeing information about the neural network hyperparameters.

# Files and Directories
This project has the following files and directories:

| File or Directory | Description |
|-------------------|-------------|
| train.csv | This CSV file contains the MNIST data set. That is, it includes the images of digits and the labels that correspond to those images. |
| Simple-Neural-Network.py | This Python file implements and runs a neural network. For more deails about the neual network configuration, see the [Neural Network](#Neural-Network) section above. |
| Neural-Network-Parameters | This directory contains te JSON files that store the neural network's metadata, nodes, and connections information. | 
| Show-Neural-Network.py | |
| static/Accuracy.png | |
| static/Cofusion.png | |
| static/Loss.png | |
| static/BadPredictions.html | |
| static/Instrumented-Neural-Networks.css | |
| Templates/index.html | |
| Templates/layout.hml | |
| tests/test_Show-Neural-Network.py | |
| tests/test_Simple-Neural-Network.py | |



