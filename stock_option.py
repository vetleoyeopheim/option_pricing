
# -*- coding: utf-8 -*-
"""
@author: Vetle
"""


import numpy as np
import pandas as pd
from numba import njit


class Option:

    """
    Parent class for different types of options
    """
    def __init__(self, series, strike, days_to_mat, init_price, 
                 est_sigma, sigma, mean, distrib = 'normal'):

        """
        Parameters
        ----------
        series : 1D numpy array (or similar)
            The historical daily returns of the underlying stock.
        strike : float
            The strike price of the option.
        days_to_mat : int
            Number of days until the option expires.
        init_price : float
            Initial price of the underlying stock.
        distrib : string, optional
            Probability distribution of the underlying stock. The default is 'normal'.
        est_sigma : boolean, optional
            Determine whether standard deviation should be estimated. If true it is estimated from the
            series given. The default is 'True'
        sigma : float, optional
            If est_sigma is False then the given sigma will be used. The default is 'None'.
        mean : float, optional
            If est_sigma is False then the given mean will be used. The default is 'None'.
        """
        
        self.series = series
        self.strike = strike
        self.days_to_mat = days_to_mat
        self.init_price = init_price
        self.distrib = distrib
        self.est_sigma = est_sigma
        self.sigma = sigma
        self.mean = mean  
        
        if self.est_sigma is False and self.sigma == None:
            print('Sigma estimation was set to false yet no sigma was given')

        if self.distrib == 'normal' and self.est_sigma is True:
            print('test')
            self.sigma = self.get_sigma()
            self.mean = self.get_mean()
        
        #TODO: Implement Kernel Density Estimation from a given data set
        elif self.distrib == 'kde':
            pass
        
        else:
            pass
    
    def get_sigma(self):
        """
        Estimate standard deviation of daily returns
        """
        return (np.std(self.series) * np.sqrt(self.days_to_mat))
    
    def get_mean(self):
        """
        Estimate mean of daily returns
        """      
        return (np.mean(self.series))
        

class EuroOption(Option):
    """
    European options are stock options that can only be exercised at maturity
    Monte carlo simulations are used to estimate the value of the option
    """
    
    def __init__(self, sim_num, series, strike, days_to_mat, int_rate,
                 init_price, est_sigma = True, sigma = None, mean = None, call = True):
        """
        Parameters
        ----------
        sim_num : int
            Number of simulation runs for the monte carlo method.
        int_rate : float
            Risk free interest rate
        """

        super().__init__(series, strike, days_to_mat, init_price, est_sigma, sigma, mean)
        self.sim_num = sim_num
        self.int_rate = int_rate
        
    def estimate_price(self):
        """
        Evaluate the value of the option for each simulation run and calculate an estimated price based on that
        """
        simulations = self.monte_carlo_sim()
        values = np.where(simulations[:, len(simulations[0]) - 1] > self.strike, simulations[:, len(simulations[0]) - 1] - self.strike, 0)
        price = np.exp(-self.int_rate * self.days_to_mat / 252) * np.(np.log(values)
        return price, values, simulations

    def monte_carlo_sim(self):
        
        simulations = np.zeros((self.sim_num, self.days_to_mat))
        price_movements = np.random.normal(loc = self.mean, scale = self.sigma, size = (self.sim_num, self.days_to_mat))
        #Set the initial value of each simulation to the initial stock price
        simulations[:, 0] = self.init_price
        #Calculate all series of the simulation
        simulations = calc_series(simulations, price_movements)
        
        return simulations
        
@njit
def calc_series(simulations, changes):
    
    for i in range(len(simulations)):
        for t in range(1,len(simulations[0])):
            simulations[i][t] = simulations[i][t - 1] * (1 + changes[i][t - 1])
            
    return simulations

a = 1
option = EuroOption(100000, np.zeros(1), 100, 100, 0.0, 105, est_sigma = False, sigma = 0.04, mean = 0.0)
price = option.estimate_price()
        