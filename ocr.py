import numpy as np
import math
import json

class OCRNeuralNetwork:
    LEARNING_RATE = 0.1
    NN_FILE_PATH = 'ocr_neural_network.json'
    
    def __init__(self, num_hidden_nodes, data_matrix, data_labels, training_indices, use_file=True):

        self._use_file = use_file
        self.data_matrix = data_matrix
        self.data_labels = data_labels
        self.training_indices = training_indices
        self.num_hidden_nodes = num_hidden_nodes
        
        if use_file:
            try:
                self._load()
                print(f"Loaded neural network from {OCRNeuralNetwork.NN_FILE_PATH}")
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                print(f"Could not load from file, initializing new network with {num_hidden_nodes} hidden nodes")
                self._init_weights(num_hidden_nodes)
        else:
            self._init_weights(num_hidden_nodes)

        if training_indices is not None and len(training_indices) > 0:
            self._train_network()
    
    def _train_network(self):
        for iteration in range(len(self.training_indices)):
            data_index = self.training_indices[iteration]
            training_data = {
                'y0': self.data_matrix[data_index],
                'label': self.data_labels[data_index]
            }
            self.train(training_data)
    
    def _rand_initialize_weights(self, size_in, size_out):
        return np.random.rand(size_out, size_in) * 0.12 - 0.06


    def _init_weights(self, num_hidden_nodes):
        self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
        self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)

        self.input_layer_bias = self._rand_initialize_weights(num_hidden_nodes, 1).T
        self.hidden_layer_bias = self._rand_initialize_weights(10, 1).T

    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.e ** -z)
    
    def sigmoid(self, z):
        vectorized_sigmoid = np.vectorize(self._sigmoid_scalar)
        return vectorized_sigmoid(z)
    
    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return np.multiply(sig, (1 - sig))

    def train(self, data):
        y0_arr = np.array(data['y0'])
        if y0_arr.ndim == 1:
            y0_arr = y0_arr.reshape(1, -1)
            
        y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(y0_arr).T)
        sum1 = y1 + np.asmatrix(self.input_layer_bias) # Add the bias
        y1 = self.sigmoid(sum1)

        y2 = np.dot(np.asmatrix(self.theta2), y1)
        y2 = np.add(y2, np.asmatrix(self.hidden_layer_bias)) # Add the bias
        y2 = self.sigmoid(y2)

        # Backpropagation
        actual_vals = [0] * 10 
        actual_vals[data['label']] = 1
        output_errors = np.asmatrix(actual_vals).T - np.asmatrix(y2)
        hidden_errors = np.multiply(np.dot(np.asmatrix(self.theta2).T, output_errors), 
                                    self.sigmoid_prime(sum1))

        # Weight Updates
        self.theta1 += self.LEARNING_RATE * np.dot(np.asmatrix(hidden_errors), 
                                                   np.asmatrix(y0_arr))
        self.theta2 += self.LEARNING_RATE * np.dot(np.asmatrix(output_errors), 
                                                   np.asmatrix(y1).T)
        self.hidden_layer_bias += self.LEARNING_RATE * np.array(output_errors)
        self.input_layer_bias += self.LEARNING_RATE * np.array(hidden_errors)

    def predict(self, test):
        test_arr = np.array(test)
        if test_arr.ndim == 1:
            test_arr = test_arr.reshape(1, -1)
            
        y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(test_arr).T)
        y1 = y1 + np.asmatrix(self.input_layer_bias) # Add the bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias) # Add the bias
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "b1": self.input_layer_bias.tolist(),
            "b2": self.hidden_layer_bias.tolist()
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH,'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = np.array(nn['theta1'])
        self.theta2 = np.array(nn['theta2'])
        self.input_layer_bias = np.array(nn['b1'])
        self.hidden_layer_bias = np.array(nn['b2'])