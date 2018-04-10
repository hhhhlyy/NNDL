import numpy as np

#parent class
class Layer:
    def __init__(self):
        pass
    def forward(self, input):
        return input
    def backward(self, input, grad_output):
        pass

class ReLU(Layer):
    def __init__(self):
        pass
    def forward(self,input):
        return np.maximum(0,input)
    def backward(self,input,grad_output):
        relu_grad = input>0
        return grad_output*relu_grad

class FC(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_units, output_units)*0.01
        self.biases = np.zeros(output_units)
    def forward(self,input):
        return np.dot(input,self.weights)+self.biases
    def backward(self,input,grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T,grad_output)/input.shape[0]
        grad_biases = grad_output.mean(axis=0)
        self.weights = self.weights - self.learning_rate*grad_weights
        self.biases = self.biases - self.learning_rate*grad_biases
        return grad_input


def forward(network, X):
    activations = []
    input = X
    for layer in network:
        activations.append(layer.forward(input))
        input = activations[-1]

    return activations

def inference(network,X):
    logits = forward(network,X)[-1]
    return logits

def train(network, X, y):
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations
    logits = layer_activations[-1]

    loss = np.square(logits - y).sum()
    loss_grad = 2.0 * (logits - y)

    for layer_i in range(len(network))[::-1]:
        layer = network[layer_i]
        loss_grad = layer.backward(layer_inputs[layer_i], loss_grad)

    return np.mean(loss)




