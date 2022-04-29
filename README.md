# Fruit-360-classification
In this project, a fully connected Artificial Neural Network(ANN) is implemented from scratch.

## Neural network architecture and details
This ANN was implemented to classify 4 classes of fruits. Feedforward algorithm was implemented in vectorized form using softmax as activation function for each layer. Back propagation was implemented in both *iterative* and *vectorized* forms with *sum of squared errors (SSE)* as cost function. *Stochastic Gradient Descent* algorithm was used to train the network.

![ANN](https://github.com/FarshidNooshi/Fruit-Detector/blob/master/assets/network.JPG)  

### Additional parts included:
- Hyperparameter tuning
- Improving SGD using momentum algorithm
- Adding more classes of fruits and hyperparameter tuning
- Using softmax as output layer's activation function  
## Dataset
[Kaggle 360-Fruits dataset](https://www.kaggle.com/moltean/fruits) was used.  
A [feature extraction and size reduction technique](https://github.com/FarshidNooshi/Fruit-Detector/blob/master/Implementation/project_assets/Feature_Extraction_Train.py) was used on train and test dataset to simplify the problem.
