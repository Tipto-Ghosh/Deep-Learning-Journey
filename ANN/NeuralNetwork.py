import numpy as np
from typing import List
from sklearn.metrics import r2_score

class NeuralNetwork:
    def __init__(self , layers: List , learning_rate: float = 0.01):
        """Initialize a Neural Network for Regression.

        Args:
            layers (List): number of neurons at each layer [input , hidden1 , ............. , output]
            learning_rate (float, optional): step size for gradient descent. Defaults to 0.01.
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.parameters = self.initialize_parameters()
    
    def fit(self , X , Y , epochs = 100 , verbose = True):
        """
        Train the network.
        """ 
        for epoch in range(epochs):
            # 1. Do the forward pass
            Y_hat , cache = self._forward(X)
            
            # 2. compute loss
            loss = self._compute_loss(Y , Y_hat)
            
            # 3. backpropagation
            gredients = self._backward(cache , Y)
            
            # 4. Upgrade paramters
            self._update_parameters(gredients)
            
            if verbose:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    def predict(self , X):
        """
        Forward pass only for prediction.
        """
        Y_hat , _ = self._forward(X)
        return Y_hat  
    
    def initialize_parameters(self):
        """ 
        Randomly initialize weights and biases for each layer.
        W(k) is the weights at k-th layer. Shape of W(k) = (neurons_in_previous_layer , neurons_in_current_layer)
        b(k) is the biases at k-th layer. Shape of b(k) = (1 , neurons_in_current_layer)
        """
        np.random.seed(42)
        parameters = {}
        
        for k in range(1 , len(self.layers)): # starting from layer index 1 cause first index is the input layer, no weights and biases.
            neurons_in_previous_layer = self.layers[k - 1]
            neurons_in_current_layer = self.layers[k]
            weight_k = np.random.randn(neurons_in_previous_layer , neurons_in_current_layer) * 0.01
            bias_k =  np.random.randn(1 , neurons_in_current_layer)
            
            # store these weights and bias in the dict.
            parameters[f'W{k}'] = weight_k
            parameters[f'b{k}'] = bias_k
        
        return parameters
    
    def _activation(self , Z):
        """Activation function: Linear Activation"""
        return Z 
    
    # --------------------- Forward Propagation ---------------------------------
    def _forward(self , X):
        """Perform forward propagation through all layers.
        Using the activation function.

        Args:
            X (_type_): Input numpy array.
        """
        cache = {
            # First layer has no activation value, k = 0 => A = X
            'A0' : X
        }    
        
        # now start finding activation values from 2nd layer 
        A_prev = X # first/input layer activation value is input itself.
        for k in range(1 , len(self.layers)):
            # get all the weights and biases of k-th layer
            W = self.parameters[f'W{k}']
            b = self.parameters[f'b{k}']
            
            # Find the value of Z(weighted sum). Z = activation_value_prev_layer . weight(k) + b(k)
            Z = np.dot(A_prev , W) + b 
            # Find the activation value
            A = self._activation(Z)
            
            # store A and Z
            cache[f'Z{k}'] = Z 
            cache[f'A{k}'] = A 
            A_prev = A 
        
        # here A is the activation value of the output layer(y_hat)
        return A , cache
    
    # -------------------------- Loss Function(MSE for Regression) --------------------
    def _compute_loss(self , Y , Y_hat): 
        """Compute MSE Loss

        Args:
            Y (_type_): actual true values.
            Y_hat (_type_): Network's predicted values.
        """
        return np.mean((Y - Y_hat) ** 2)
    
    # ----------------- Backward Propagation -----------------
    def _backward(self , cache , Y):
        """ 
        Perform backpropagation to compute gradients for all layers.
        Since activation is linea, f'(Z) = 1
        """
        m = Y.shape[0]
        grads = {} # store the gradients of all weights and biases for all layers.
        L = len(self.layers) - 1 # Last layer index.
        
        # Step-1: compute dA for last layer from loss derivative for batch size m
        # dJ/dA^(L) = - 2/m * (Y - A^(L)) or - 2/m * (Y - Y_hat)
        Y_hat = cache[f'A{L}'] # last layer's activation value
        
        dA = (-2/m) * (Y - Y_hat)
        
        # Step-2: Loop backward from 2nd last to first layer
        for k in reversed(range(1 , len(self.layers))):
            A_prev = cache[f'A{k - 1}']
            
            # Linear Activation function => f'(Z) = 1 => dA = dZ
            dZ = dA
            
            # We need to find the derivatives of loss w.r.t weights and biases
            dW = np.dot(A_prev.T , dZ)
            db = np.sum(dZ , axis = 0 , keepdims = True)
            
            # store the gradient for update weights and biases later
            grads[f'dW{k}'] = dW
            grads[f'db{k}'] = db
            
            # propogate gradient to previous layer's activation
            W = self.parameters[f'W{k}']
            dA = np.dot(dZ , W.T)
        
        return grads
    
    # ------------------------ Upgrade Weights and Biases ------------------------
    def _update_parameters(self , gradients):
        """ 
        Apply Gradient Descent step for all layers.
        W := W - lr * dW
        b := b - lr * db
        """  
        
        for k in range(1 , len(self.layers)):
            self.parameters[f'W{k}'] -= self.learning_rate * gradients[f'dW{k}']
            self.parameters[f'b{k}'] -= self.learning_rate * gradients[f'db{k}']
            
        

        

X = np.array([[1], [2], [3], [4], [5]])
Y = np.array([[2], [4], [6], [8], [10]])

nn = NeuralNetwork(layers = [1, 3, 1] , learning_rate = 0.01)

nn.fit(X, Y, epochs = 10)

preds = nn.predict(X)
print("\nPredictions:\n", preds)

print("R2 score: " , r2_score(Y , preds))