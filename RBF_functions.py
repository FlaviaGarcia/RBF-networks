# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:59:25 2020

@author: flaviagv
"""


import numpy as np 
import matplotlib.pyplot as plt




class RadialBasisFunctions():
    
    def __init__(self, n_nodes, sigma_RBF_nodes, learning_rate):
        self.n_nodes = n_nodes
        self.learning_rate = learning_rate
        self.sigma_RBF_nodes = sigma_RBF_nodes
        self.mu_RBF_nodes = None
        self.w = None

    
    def build_phi(self, x_train):

        n_training_samples = x_train.shape[0]
        
        phi = np.zeros((n_training_samples, self.n_nodes))
        
        for idx_rbf_node in range(self.n_nodes):
            phi[:,idx_rbf_node] = self.gauss_transfer_function(x_train, idx_rbf_node)     
        return phi 
    
    
    def gauss_transfer_function(self, x_train, idx_rbf_node):
        # square euclidean distance 
        r_square = np.square(self.compute_euclidean_distance(x_train, idx_rbf_node))
        gauss_transfer_function_train_samples = np.exp(-1*np.divide(r_square, (2*np.square(self.sigma_RBF_nodes))))
        return gauss_transfer_function_train_samples
    
    
    def compute_euclidean_distance(self, x_train, idx_rbf_node):
        mu_RBF_node = self.mu_RBF_nodes[idx_rbf_node]
        sqrt_term = np.sum(np.square(x_train - mu_RBF_node), axis = 1)
        euclidean_distance = np.sqrt(sqrt_term)
        return euclidean_distance
    
  
    def least_squares(self, phi, f):
        self.w = np.linalg.inv(np.transpose(phi) @ phi) @ np.transpose(phi) @ f

    
    def delta_learning(self, f_train, phi_train, epochs, randomize_samples=False):
        # Initialize weights randomly from a normal distribution
        dimensions_training_data = f_train.shape[1]
        self.w = np.random.randn(self.n_nodes, dimensions_training_data).reshape(-1, dimensions_training_data)
        
        # Iteratively call delta rule function and update weights
        for i in range(epochs):
            if randomize_samples:
                rand_ids = np.random.permutation(f_train.shape[0])     
            else:
                rand_ids = range(f_train.shape[0])
                
            for idx_training_point in rand_ids:
                f_point = f_train[idx_training_point]
                f_point= f_point.reshape(1,-1)
                self.w += self.delta_rule(f_point, phi_train[idx_training_point])
        
    
    def delta_rule(self, f_point, phi_point):
        
        if f_point.shape[0] != 1:
            raise Exception("Only pass one point at at time to delta_rule func")
        
        # Calculate weight updates (column vector)
        error = f_point - np.dot(phi_point, self.w)
        dw = self.learning_rate * error * phi_point.reshape(-1,1)
        return dw
    
    
    def competitive_learning(self, x_train, n_epochs):
        # initialize RBF nodes to data samples 
        self.init_mu_RBF_from_data_points(x_train)
        init_mu_RBF_nodes = self.mu_RBF_nodes.copy()
        for epoch in range(n_epochs):
            idx_training_sample = np.random.randint(0, x_train.shape[0]-1)
            x_train_point = x_train[idx_training_sample]

            self.update_mu_CL(x_train_point)
        return init_mu_RBF_nodes        
    
    
    def init_mu_RBF_from_data_points(self, x_train):
        idx_points_for_mu = np.random.choice(x_train.shape[0], self.n_nodes)  
        self.mu_RBF_nodes = x_train[idx_points_for_mu]
  
    
    def update_mu_CL(self, x_train_point):
        """
        Update the mu of the winning RBF node (the one most near x_training_point)
        """
        idx_winning_rbf = np.argsort(self.compute_euclidean_distance(x_train_point, np.arange(self.n_nodes)))[0]
        # Update winner [shift towards x]
        self.mu_RBF_nodes[idx_winning_rbf] += self.learning_rate * (x_train_point - self.mu_RBF_nodes[idx_winning_rbf])
 
        
    
    # Calculate mean square error between function and its approximation
    def MSE(self, f, f_hat):
        return np.mean(np.square(f - f_hat))
    
    # Calculate absolute residual error between function and its approximation
    def ARE(self, f, f_hat):
        return np.mean(np.abs(f - f_hat))
    
     
    def plot_pred_results(self, x, f_x, fhat_x, use_CL, mu_vec_init, train_or_test):
        text_title = str(self.n_nodes) + " hidden nodes over " + train_or_test + " data"

        plt.plot(x, f_x,'k',label='Real')
        plt.plot(x, fhat_x, '--c', label='Predicted')
        if use_CL:
            plt.scatter(mu_vec_init, np.zeros(len(mu_vec_init)), label="initial mu") 
            plt.scatter(self.mu_RBF_nodes.reshape(-1,), np.zeros(len(self.mu_RBF_nodes)), label="mu after CL", marker="x", c="r",) 
        else:
            plt.scatter(self.mu_RBF_nodes.reshape(-1,), np.zeros(len(self.mu_RBF_nodes)), marker="x", c="r", label="RBF centers")

        plt.legend()
        plt.title(text_title)
        plt.show()
        


    def predict(self, phi):
        return np.dot(phi, self.w)
