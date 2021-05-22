from __future__ import division
import multiprocessing
import itertools
import numpy as np
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt

vec0 = np.array([[1], [0]])
vec1 = np.array([[0], [1]])


def prob_0(phi):
    return 0.25*(1.+np.cos(phi))


def prob_1(phi):
    return 0.25*(1.-np.cos(phi))


def prob_2(phi):
    return 0.25*(1.+np.sin(phi))


def prob_3(phi):
    return 0.25*(1.-np.sin(phi))


def state_vec(phi):
    return (1./np.sqrt(2))*(vec0+np.exp(-1.j*phi)*vec1)


def cost(real_value, estimator):
    return 4*np.sin((real_value-estimator)/2.)**2


def cost_optimal(num_measurements):
    with np.errstate(divide='raise'):
        max_cost = 2
        try:
            return_candidate = 2 - (1/pow(2, num_measurements-1))*np.sum([np.sqrt(special.binom(num_measurements, n)*special.binom(num_measurements, n-1)) for n in range(num_measurements)])
            if return_candidate <= max_cost:
                return return_candidate
            else:
                return 1/num_measurements
        except FloatingPointError:
            return 1/num_measurements


def maximum_likelihood_func(n0, n1, n2, n3, phi):
    return n0*np.log(prob_0(phi)) + n1*np.log(prob_1(phi)) + n2*np.log(prob_2(phi)) + n3*np.log(prob_3(phi))


def maximum_likelihood_func_minus1(n0, n1, n2, n3):
    def func(phi):
        return (-1)*maximum_likelihood_func(n0, n1, n2, n3, phi)
    return func


def run_simulation(parameter_values, num_experiments):
    costs_arr = []
    for parameter_value in parameter_values:
        with np.errstate(divide='raise'):
            try:
                ns_arr = np.random.multinomial(num_experiments, [prob_0(parameter_value), prob_1(parameter_value), prob_2(parameter_value), prob_3(parameter_value)])
                parameter_estimator = optimize.fmin(maximum_likelihood_func_minus1(ns_arr[0], ns_arr[1], ns_arr[2], ns_arr[3]), x0=[parameter_value])[0]
                costs_arr.append(cost(parameter_value, parameter_estimator))
            except FloatingPointError:
                continue
    if costs_arr:
        average_cost = sum(costs_arr)/len(costs_arr)
    return num_experiments, average_cost


def main():
    max_num_experiments = 1000  # a.k.a. N
    num_experiments_step = 10
    num_parameter_values = 10000  # shall be evenly spaced in [0,2*Pi]

    num_experiments_array = np.arange(3, max_num_experiments, num_experiments_step)
    parameter_values = np.arange(0, 2*np.pi, 2*np.pi/num_parameter_values)

    args_tuples = tuple(zip(itertools.repeat(parameter_values, len(num_experiments_array)), num_experiments_array))
    pool = multiprocessing.Pool(processes=4)
    simulation_results = pool.starmap(run_simulation, args_tuples)
    pool.close()
    pool.join()
    num_measurements_optimal_cost = []
    optimal_cost_arr = []
    print(simulation_results)
    for measurementn in np.arange(2, max_num_experiments, 1):
        num_measurements_optimal_cost.append(measurementn)
        optimal_cost_arr.append(cost_optimal(measurementn))

    fig = plt.figure()
    ax = plt.gca()
    (num_exp, cost_estim) = zip(*simulation_results)
    ax.scatter(num_exp, cost_estim, c='red')
    ax.plot(num_measurements_optimal_cost, optimal_cost_arr)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
    return

if __name__ == '__main__':
    main()