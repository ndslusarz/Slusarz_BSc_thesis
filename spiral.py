# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:38:19 2021

@author: Natalia Ślusarz
"""

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np




'''class for the spiral dataset'''


class Data_spiral:
    
    '''
    takes entanglemtnt, min_phi, number of samples and maximum radius of the spiral
    '''
    def __init__(self, ent, min_phi, num_samples,radius):
        self.ent = ent #entanglment measure
        self.min_phi = min_phi; #should be >0 to avoid clutter at origin
        self.max_phi = min_phi + self.ent 
        self.num_samples = num_samples
        self.radius = radius
       
    #helper function for data creation
    def gen_x_axis_values(self, rot_angles, swap_val=1):
        x_values = self.radius * rot_angles * np.cos(rot_angles) * swap_val
        return x_values.reshape((-1, 1))
    
    #helper function for data creation
    def gen_y_axis_values(self, rot_angles, swap_val=1):
        y_values = self.radius * rot_angles * np.sin(rot_angles) * swap_val
        return y_values.reshape((-1, 1))    
    
    '''
   generating data
    
    '''
    def generate_data(self):
        self.rot_angles = np.linspace(start=self.min_phi, stop=self.max_phi, num=self.num_samples)
        
        # Data for class 0:
        x1_values_0 = self.gen_x_axis_values(
             rot_angles=self.rot_angles)
        x2_values_0 = self.gen_y_axis_values(
             rot_angles=self.rot_angles)
        x_data_0 = np.hstack((x1_values_0, x2_values_0))
        y_data_0 = np.zeros(shape=(self.num_samples, 1))
        
        # Data for class 1:
        x1_values_1 = self.gen_x_axis_values(
             rot_angles=self.rot_angles, swap_val=-1)
        x2_values_1 = self.gen_y_axis_values(
             rot_angles=self.rot_angles, swap_val=-1)
        x_data_1 = np.hstack((x1_values_1, x2_values_1))
        y_data_1 = np.ones(shape=(self.num_samples, 1))
        
        # Stack to create data matrix:
        self.x_data = np.vstack((x_data_0, x_data_1))
        self.y_data = np.vstack((y_data_0, y_data_1))
        
    '''
    generating data positioned slightly off the original spiral (random noise)
    max_dist_away - controls how far from the spiral points can be
    '''

    def generate_adjacant_data(self, max_dist_away):
        self.rot_angles = np.linspace(start=self.min_phi, stop=self.max_phi, num=self.num_samples)
        
        # Data for class 0:
        x1_values_0 = self.gen_x_axis_values(
             rot_angles=self.rot_angles)
        x2_values_0 = self.gen_y_axis_values(
             rot_angles=self.rot_angles)
        x_data_0 = np.hstack((x1_values_0, x2_values_0))
        y_data_0 = np.zeros(shape=(self.num_samples, 1))
        
        # Data for class 1:
        x1_values_1 = self.gen_x_axis_values(
             rot_angles=self.rot_angles, swap_val=-1)
        x2_values_1 = self.gen_y_axis_values(
             rot_angles=self.rot_angles, swap_val=-1)
        x_data_1 = np.hstack((x1_values_1, x2_values_1))
        y_data_1 = np.ones(shape=(self.num_samples, 1))
        
        # Stack to create data matrix:
        self.x_data = np.vstack((x_data_0, x_data_1))
        self.y_data = np.vstack((y_data_0, y_data_1))
        
    
        for i in range(len(self.x_data)):
            diff = np.random.uniform(-max_dist_away, max_dist_away)
            self.x_data[i][0] += diff
            diff = np.random.uniform(-max_dist_away, max_dist_away)
            self.x_data[i][1] += diff
         
    '''
    visualises the spiral as a graph
    '''
    def visualise(self):
        plt.figure()
        plt.figure(figsize=(5,5), dpi=600)
        plt.scatter(
        self.x_data[:,0], self.x_data[:,1], c=self.y_data, cmap='RdBu')
        plt.title('Binary spiral dataset in 2d')
        plt.show()

