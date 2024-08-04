import json
import numpy as np

def load_data():
    file_1 = open('datasets\Test_data', "r")
    file_2 = open('datasets\Training_data', "r")
    data_TEST = json.load(file_1)
    data_TRAIN = json.load(file_2)
    file_1.close()
    file_2.close()
    P_data_TEST = data_TEST["Pic"]
    N_data_TEST = data_TEST["letter_num"]
    P_data_TRAIN = data_TRAIN["Pic"]
    N_data_TRAIN = data_TRAIN["letter_num"]
    return P_data_TEST, N_data_TEST, P_data_TRAIN, N_data_TRAIN

def load_dataset():
    P_data_TEST, N_data_TEST, P_data_TRAIN, N_data_TRAIN = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in P_data_TRAIN]
    training_results = [vectorized_result(y) for y in N_data_TRAIN]
    training_data = list(zip(training_inputs, training_results))
    #print(len(training_data))

    #validation_inputs = [np.reshape(x, (784, 1)) for x in P_data_TEST]
    #validation_data = list(zip(validation_inputs, N_data_TEST))
    #print(len(validation_data))

    test_inputs = [np.reshape(x, (784, 1)) for x in P_data_TEST]
    test_data = list(zip(test_inputs, N_data_TEST))
    #print(len(test_data))
    
    #return (training_data, validation_data, test_data)
    return (training_data, test_data)

def vectorized_result(j):
    e = np.zeros((33, 1))
    e[j] = 1.0
    return e