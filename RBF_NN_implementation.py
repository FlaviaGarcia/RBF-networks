# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:35:36 2020

@author: flaviagv
"""

import math
import numpy as np 
import matplotlib.pyplot as plt
from RBF_functions import RadialBasisFunctions
import argparse


def RBF_NN_batch_learning(n_nodes, learning_rate, sin_or_square, sigma_RBF_nodes, batch_or_online_learning, use_CL, n_epochs_online_learning=0, n_epochs_CL=0, plot_train_results=False, plot_test_results=True, verbose=True, add_noise = False, train_range=[0,2*math.pi], test_range=[0.05,2*math.pi+0.05], step=0.1):
    
    x_train, f_train = generate_sin_and_square(train_range, step, sin_or_square, add_noise)
    x_test, f_test = generate_sin_and_square(test_range, step, sin_or_square, add_noise)
    
    rbf = RadialBasisFunctions(n_nodes, sigma_RBF_nodes, learning_rate)
    
    if use_CL:
        init_mu_RBF_nodes = rbf.competitive_learning(x_train, n_epochs_CL)
    else:
        rbf.mu_RBF_nodes = generate_equally_distrib_mu(n_nodes)
        init_mu_RBF_nodes = rbf.mu_RBF_nodes.copy()
    
    
    
    # Build phi arrays: RBF node output
    phi_train = rbf.build_phi(x_train)
    phi_test = rbf.build_phi(x_test)     
    
    if batch_or_online_learning == "batch":
        rbf.least_squares(phi_train, f_train)
    elif batch_or_online_learning == "online":
        rbf.delta_learning(f_train, phi_train, n_epochs_online_learning, randomize_samples=True)
    else:
        raise ValueError("batch_or_online_learning can just have as values 'batch' or 'online'")

    
    fhat_train = rbf.predict(phi_train)
    fhat_test = rbf.predict(phi_test)
    
    ARE_train = rbf.ARE(f_train, fhat_train)
    ARE_test = rbf.ARE(f_test, fhat_test)
    
    if verbose:
        print("Training ARE: ", ARE_train)
        print("Training visual fit")
        
        print("Testing ARE: ", ARE_test)
        print("Testing visual fit")
        
    if plot_train_results:
        rbf.plot_pred_results(x_train, f_train, fhat_train, use_CL, init_mu_RBF_nodes, train_or_test = "train")
        
    if plot_test_results:
        rbf.plot_pred_results(x_test, f_test, fhat_test, use_CL, init_mu_RBF_nodes, train_or_test = "test")
        
    return ARE_test


def generate_equally_distrib_mu(n_nodes, mu_range = [0, round(2*math.pi,1)]):
    mu_RBF_nodes = np.linspace(mu_range[0], mu_range[1], n_nodes) #rows=number of RBF nodes, cols=number of dimensions
    n_nodes = mu_RBF_nodes.reshape(len(mu_RBF_nodes),-1)
    return n_nodes


def generate_sin_and_square(xrange, step, sin_or_square, add_noise, mu_noise=0, std_noise=0.01, plot_dataset=False):
    x = np.arange(xrange[0],xrange[1],step)
    sin_f_x = np.sin(2*x)
    if sin_or_square == "sin":
        f_x = sin_f_x     
    elif sin_or_square == "square":
        f_x = np.sign(sin_f_x)
    
    if add_noise:
        f_x = add_gaussian_noise(f_x, mu_noise, std_noise)
    
    if plot_dataset: 
        plt.plot(x,f_x)
        plt.show()
        
    # reshape to have rows = n_training samples and cols= n_dimensions
    x = x.reshape(len(x),-1)
    f_x = f_x.reshape(len(f_x),-1)

    return x, f_x


def add_gaussian_noise(x, mu, std):
    noise = np.random.normal(mu, std, x.size)
    return x + noise







if __name__ == "__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument('--n_nodes', type= int, default=11)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--sin_or_square', type=str, default="sin")
    parser.add_argument('--sigma_RBF_nodes', type=float, default=1.0)
    parser.add_argument('--batch_or_online_learning', type= str, default="batch")
    parser.add_argument('--n_epochs_online_learning', type=int, default=10000)
    parser.add_argument('--use_CL', type=bool, default=False)
    parser.add_argument('--n_epochs_CL', type=int, default=1000)
    parser.add_argument('--plot_train_results', type=bool, default=False)
    parser.add_argument('--plot_test_results', type=bool, default=True)
    parser.add_argument('--verbose', type=int, default=1)

    args=parser.parse_args()
        
    RBF_NN_batch_learning(args.n_nodes, args.learning_rate, args.sin_or_square, args.sigma_RBF_nodes, args.batch_or_online_learning, args.use_CL, args.n_epochs_online_learning, args.n_epochs_CL, args.plot_train_results, args.plot_test_results, args.verbose)


