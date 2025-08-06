import numpy as np
import csv
from functions import *
import matplotlib.pyplot as plt

class Network(object):
    def __init__(self):
        
        # adam variables
        self.first_call = True
        self.learning_rate = 0.0001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.time_step = 0 

        # adam initialized parameters
        self.m_output_weights = None
        self.v_output_weights = None
        self.m_output_bias = None
        self.v_output_bias = None
        self.m_hidden_weights = None
        self.v_hidden_weights = None
        self.m_hidden_bias = None
        self.v_hidden_bias = None
        
        
        # hyperparameters        
        self.input_neurons = 784
        self.hidden_neurons = 128
        self.output_neurons = 10 
        
        # He Initialization
        # variance * shape(x, y)
        self.input_to_hidden_weight_matrix = np.sqrt(2/self.input_neurons) * np.random.randn(self.input_neurons, self.hidden_neurons)
        self.hidden_to_output_weight_matrix = np.sqrt(2/self.hidden_neurons) * np.random.randn(self.hidden_neurons, self.output_neurons)            
            
            
        self.input_to_hidden_biases = np.zeros(self.hidden_neurons)
        self.hidden_to_output_biases = np.zeros(self.output_neurons)
        
    
                
    def forward_prop(self, x):
        
        z1 = x @ self.input_to_hidden_weight_matrix + self.input_to_hidden_biases
        a1 = relu(z1)
        
        z2 = a1 @ self.hidden_to_output_weight_matrix + self.hidden_to_output_biases
        
        return z1, a1, z2   
    

    def loss_compute(self, z2, y):
        
        stable_z = z2 - np.max(z2)
        
        exp_z = np.exp(stable_z)
        prediction_softmax_vector = exp_z / np.sum(exp_z)

        epsilon = 1e-15
        
        prediction_softmax_vector = np.clip(prediction_softmax_vector, epsilon, 1 - epsilon)
        cross_entropy = -np.sum(y * np.log(prediction_softmax_vector))

        nabla_loss_wrt_a = prediction_softmax_vector - y
        
        return cross_entropy, nabla_loss_wrt_a

    def back_prop(self, nabla_loss_wrt_a, z1, a1, x):
        
        output_layer_error = nabla_loss_wrt_a
        
        
        
        delta_output_loss_bias = output_layer_error
        
        delta_output_loss_weights = np.outer(a1, output_layer_error)
        
        
        propagated_error = output_layer_error @ self.hidden_to_output_weight_matrix.T
        
        hidden_layer_error = propagated_error * relu_prime(z1)

        
        delta_hidden_loss_bias = hidden_layer_error
        
        delta_hidden_loss_weights = np.outer(x, hidden_layer_error)
   
        
        return delta_output_loss_bias, delta_output_loss_weights, delta_hidden_loss_bias, delta_hidden_loss_weights
    
    
    def grad(self, delta_output_loss_bias, delta_output_loss_weights, delta_hidden_loss_bias, delta_hidden_loss_weights):
        
        if self.first_call:
            self.m_output_weights = np.zeros_like(delta_output_loss_weights)
            self.v_output_weights = np.zeros_like(delta_output_loss_weights)
            
            self.m_output_bias = np.zeros_like(delta_output_loss_bias)
            self.v_output_bias = np.zeros_like(delta_output_loss_bias)

            self.m_hidden_weights = np.zeros_like(delta_hidden_loss_weights)
            self.v_hidden_weights = np.zeros_like(delta_hidden_loss_weights)
            
            self.m_hidden_bias = np.zeros_like(delta_hidden_loss_bias)
            self.v_hidden_bias = np.zeros_like(delta_hidden_loss_bias)
            
            self.first_call = False
            
        self.time_step += 1
        
            
        # compute biased raw estimates
        self.m_output_weights = self.beta_1 * self.m_output_weights + (1-self.beta_1) * delta_output_loss_weights
        self.m_output_bias = self.beta_1 * self.m_output_bias + (1-self.beta_1) * delta_output_loss_bias
        
        self.v_output_weights = self.beta_2 * self.v_output_weights + (1-self.beta_2) * (delta_output_loss_weights**2)
        self.v_output_bias = self.beta_2 * self.v_output_bias + (1-self.beta_2) * (delta_output_loss_bias**2)
        
        self.m_hidden_weights = self.beta_1 * self.m_hidden_weights + (1-self.beta_1) * delta_hidden_loss_weights
        self.m_hidden_bias = self.beta_1 * self.m_hidden_bias + (1-self.beta_1) * delta_hidden_loss_bias

        self.v_hidden_weights = self.beta_2 * self.v_hidden_weights + (1-self.beta_2) * (delta_hidden_loss_weights**2)
        self.v_hidden_bias = self.beta_2 * self.v_hidden_bias + (1-self.beta_2) * (delta_hidden_loss_bias**2)
        
        # compute hat variables (bias-corrected)
        m_hat_output_weights = self.m_output_weights / (1-self.beta_1**self.time_step)
        m_hat_output_bias = self.m_output_bias / (1-self.beta_1**self.time_step)
        
        m_hat_hidden_weights = self.m_hidden_weights / (1-self.beta_1**self.time_step)
        m_hat_hidden_bias = self.m_hidden_bias / (1-self.beta_1**self.time_step)


        v_hat_output_weights = self.v_output_weights / (1-self.beta_2**self.time_step)
        v_hat_output_bias = self.v_output_bias / (1-self.beta_2**self.time_step)
        
        v_hat_hidden_weights = self.v_hidden_weights / (1-self.beta_2**self.time_step)
        v_hat_hidden_bias = self.v_hidden_bias / (1-self.beta_2**self.time_step)
        
        # update parameters 
        self.input_to_hidden_weight_matrix = self.input_to_hidden_weight_matrix - self.learning_rate * m_hat_hidden_weights / (np.sqrt(v_hat_hidden_weights) + self.epsilon)
        self.input_to_hidden_biases = self.input_to_hidden_biases - self.learning_rate * m_hat_hidden_bias / (np.sqrt(v_hat_hidden_bias) + self.epsilon)
        
        self.hidden_to_output_weight_matrix = self.hidden_to_output_weight_matrix - self.learning_rate * m_hat_output_weights/(np.sqrt(v_hat_output_weights) + self.epsilon)
        self.hidden_to_output_biases = self.hidden_to_output_biases - self.learning_rate * m_hat_output_bias / (np.sqrt(v_hat_output_bias) + self.epsilon)

    
    
net = Network()
epochs = 5
training_loss = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_losses = []
    
    with open('mnist_train.csv', 'r') as data:
        dataset = csv.reader(data)
        next(dataset)  # Skip header

        for i, row in enumerate(dataset):
            y = row[0]
            y_true = np.zeros(10, dtype='float64')
            y_true[int(y)] = 1.0
            
            x = np.array(row[1:], dtype='float64')
            x = x / 255.0  # Normalize input

            z1, a1, z2 = net.forward_prop(x)
            
            cross_entropy, nabla_loss_wrt_a = net.loss_compute(z2, y_true)

            (
                delta_output_loss_bias, 
                delta_output_loss_weights, 
                delta_hidden_loss_bias, 
                delta_hidden_loss_weights
            ) = net.back_prop(nabla_loss_wrt_a, z1, a1, x)

            
            net.grad(delta_output_loss_bias, delta_output_loss_weights, 
                     delta_hidden_loss_bias, delta_hidden_loss_weights)
            
            epoch_losses.append(cross_entropy)

            if i % 1000 == 0:
                print(f"[Sample {i}] Intermediate Loss: {cross_entropy:.6f}")
    
    avg_loss = np.mean(epoch_losses)
    training_loss.append(avg_loss)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")

            
