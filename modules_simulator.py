import numpy as np, keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import math, sys
#from matplotlib import pyplot as plt
from random import randint
#plt.switch_backend('Qt4Agg')
import keras



def init_model(num_hnodes = 35):
    model = Sequential()
    model.add(Dense(num_hnodes, input_dim=21, init='he_uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(19,init='he_uniform'))
    model.compile(loss='mean_absolute_error', optimizer='Nadam')
    return model



def data_preprocess(filename = 'ColdAir.csv', downsample_rate=25):

     #Import training data and clear away the two top lines
    train_data = np.loadtxt(filename, delimiter=',', skiprows= 2 )

    #Splice data (downsample)
    ignore = np.copy(train_data)
    train_data = train_data[0::downsample_rate]
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[1]):
            if ( i != train_data.shape[0]-1):
                train_data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,j].sum()/downsample_rate
            else:
                residue = ignore.shape[0] - i * downsample_rate
                train_data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue,j].sum()/residue

    #Normalize between 0-0.99
    normalizer = np.zeros(train_data.shape[1], dtype=np.float64)
    min = np.zeros(len(train_data[0]), dtype=np.float64)
    max = np.zeros(len(train_data[0]), dtype=np.float64)
    for i in range(len(train_data[0])):
        min[i] = np.amin(train_data[:,i])
        max[i] = np.amax(train_data[:,i])
        normalizer[i] = max[i]-min[i] + 0.00001
        train_data[:,i] = (train_data[:,i] - min[i])/ normalizer[i]

    return train_data, max, min







def novelty(weak_matrix, archive, k = 10):
    import bottleneck
    #Handle early gens with archive size less that 10
    if (len(archive) < k):
        k = len(archive)

    novel_matrix = np.zeros(len(archive))
    for i in range(len(archive)):
        novel_matrix[i] = np.sum(np.square(weak_matrix - archive[i]))

    #k-nearest neighbour algorithm
    k_neigh = bottleneck.partsort(novel_matrix, k)[:k] #Returns a subarray of k smallest novelty scores

    #Return novelty score as the average Euclidean distance (behavior space) between its k-nearest neighbours
    return np.sum(k_neigh)/k

def get_model_arch(seed = 'Evolutionary/seed.json'): #Get model architecture
    import json
    from keras.models import model_from_json
    with open(seed) as json_file:
        json_data = json.load(json_file)
    model_arch = model_from_json(json_data)
    return model_arch

def print_results(model_name, is_reccurrent = False, filename = 'ColdAir.csv', downsample_rate=25, seed = 'Evolutionary/seed.json', n_prev = 7):
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

    if not is_reccurrent:
        n_prev = 1
    #Load model
    model = get_model_arch(seed)
    model.load_weights(model_name)
    model.compile(loss='mse', optimizer='rmsprop')

    #Import training data and clear away the two top lines
    train_data = np.loadtxt(filename, delimiter=',', skiprows= 2 )

    #Splice data (downsample)
    ignore = np.copy(train_data)
    train_data = train_data[0::downsample_rate]
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[1]):
            if ( i != train_data.shape[0]-1):
                train_data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,j].sum()/downsample_rate
            else:
                residue = ignore.shape[0] - i * downsample_rate
                train_data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue,j].sum()/residue


    #Normalize between 0-0.99
    normalizer = np.zeros(len(train_data[0]), dtype=np.float64)
    min = np.zeros(len(train_data[0]), dtype=np.float64)
    max = np.zeros(len(train_data[0]), dtype=np.float64)
    for i in range(len(train_data[0])):
        min[i] = np.amin(train_data[:,i])
        max[i] = np.amax(train_data[:,i])
        normalizer[i] = max[i]-min[i] + 0.00001
        train_data[:,i] = (train_data[:,i] - min[i])/ normalizer[i]

    ##EO FILE IO##

    print ('TESTING NOW')
    if is_reccurrent:
        input = np.reshape(train_data[0:n_prev], (1, n_prev, 21))  # First training example in its entirety
        track_time_target = np.reshape(np.zeros((len(train_data)-n_prev) * 19), (19, len(train_data)-n_prev))
        track_time_output = np.reshape(np.zeros((len(train_data)-n_prev) * 19), (19, len(train_data)-n_prev))
    else:
        input = np.reshape(train_data[0], (1, 21))  # First input to the simulatior
        track_time_target = np.reshape(np.zeros((len(train_data) - 1) * 19), (19, len(train_data) - 1))
        track_time_output = np.reshape(np.zeros((len(train_data) - 1) * 19), (19, len(train_data) - 1))

    for example in range(len(train_data) - n_prev):  # For all training examples
        model_out = model.predict(input)  # Time domain simulation

        # Track index
        for index in range(19):
            track_time_output[index][example] = model_out[0][index] * normalizer[index] + min[index]
            track_time_target[index][example] = train_data[example + n_prev][index] * normalizer[index] + min[index]

        # Fill in new input data
        if is_reccurrent:
            for k in range(len(model_out)):  # Modify the last slot
                input[0][0][k] = model_out[0][k]
                input[0][0][k] = model_out[0][k]
            # Fill in two control variables
            input[0][0][19] = train_data[example + n_prev][19]
            input[0][0][20] = train_data[example + n_prev][20]
            input = np.roll(input, -1, axis=1)  # Track back everything one step and move last one to the last row
        else:
            for k in range(len(model_out)):
                input[0][k] = model_out[0][k]
                input[0][k] = model_out[0][k]
            # Fill in two control variables
            input[0][19] = train_data[example + 1][19]
            input[0][20] = train_data[example + 1][20]


    #ignore = abs(track_time_target - track_time_output)
    for index in range(19):
        #plt.plot(ignore[index], 'r--',label='Target Index: ' + str(index))
        plt.plot(track_time_target[index], 'r--',label='Actual Data: ' + str(index))
        plt.plot(track_time_output[index], 'b-',label='Recurrent Simulator: ' + str(index))
        plt.legend( loc='upper right',prop={'size':6})
        #plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
        plt.show()


def metrics(train_data, model, max, min, time_domain = True, filename = 'ColdAir.csv', downsample_rate=25, noise = False, noise_mag = 0.1):
    #Load model
    #model = theanets.Regressor.load(model_name)
    input = np.reshape(train_data[0], (1, 21)) #First input to the simulatior

    #Track across time steps
    track_time_target = np.reshape(np.zeros((len(train_data)-1) * 19), (19, len(train_data)-1))
    track_time_output = np.reshape(np.zeros((len(train_data)-1) * 19), (19, len(train_data)-1))
    error = np.reshape(np.zeros((len(train_data)-1) * 19), (19, len(train_data)-1))

    for example in range(len(train_data)-1):#For all training examples
        if (time_domain):
            model_out = model.predict(input) #Time domaain simulation
        else:
             model_out = model.predict(np.reshape(train_data[example], (1, 21))) #Non-domain simulation
        #Track index
        for index in range(19):
            track_time_output[index][example] = model_out[0][index] * (max[index]-min[index] + 0.00001) + min[index]
            track_time_target[index][example] = train_data[example+1][index] * (max[index]-min[index] + 0.00001) + min[index]
        #Fill in new input data
        for k in range(len(model_out)):
            input[0][k] = model_out[0][k]
            input[0][k] = model_out[0][k]
        #Fill in two control variables
        input[0][19] = train_data[example+1][19]
        input[0][20] = train_data[example+1][20]
    error = abs(track_time_output - track_time_target)
    for index in range(19):
        error[index][:] = error[index][:] * 100 / max[index]
    return error

def return_results(time_domain = True, model_name = 'Evolutionary/temp/0',filename = 'ColdAir.csv', downsample_rate=25):
    #Load model
    import theanets
    model = theanets.Regressor.load(model_name)

    #Import training data and clear away the two top lines
    train_data = np.loadtxt(filename, delimiter=',', skiprows= 2 )

    #Splice data (downsample)
    ignore = np.copy(train_data)
    train_data = train_data[0::downsample_rate]
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[1]):
            if ( i != train_data.shape[0]-1):
                train_data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,j].sum()/downsample_rate
            else:
                residue = ignore.shape[0] - i * downsample_rate
                train_data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue,j].sum()/residue


    #Normalize between 0-0.99
    normalizer = np.zeros(len(train_data[0]), dtype=np.float64)
    min = np.zeros(len(train_data[0]), dtype=np.float64)
    max = np.zeros(len(train_data[0]), dtype=np.float64)
    for i in range(len(train_data[0])):
        min[i] = np.amin(train_data[:,i])
        max[i] = np.amax(train_data[:,i])
        normalizer[i] = max[i]-min[i] + 0.00001
        train_data[:,i] = (train_data[:,i] - min[i])/ normalizer[i]

    ##EO FILE IO##

    print ('TESTING NOW')
    input = np.reshape(train_data[0], (1, 21)) #First input to the simulatior
    error = np.zeros(21) #Error array to track error for each variables

    #Track across time steps
    track_time_target = np.reshape(np.zeros((len(train_data)-1) * 19), (19, len(train_data)-1))
    track_time_output = np.reshape(np.zeros((len(train_data)-1) * 19), (19, len(train_data)-1))

    for example in range(len(train_data)-1):#For all training examples

        if (time_domain):
            model_out = model.predict(input) #Time domaain simulation
        else:
             model_out = model.predict(np.reshape(train_data[example], (1, 21))) #Non-domain simulation

        #Track index
        for index in range(19):
            track_time_output[index][example] = model_out[0][index] * normalizer[index] + min[index]
            track_time_target[index][example] = train_data[example+1][index] * normalizer[index] + min[index]

        #Fill in new input data
        for k in range(len(model_out)):
            input[0][k] = model_out[0][k]
            input[0][k] = model_out[0][k]
        #Fill in two control variables
        input[0][19] = train_data[example+1][19]
        input[0][20] = train_data[example+1][20]


    list = []
    list.append(track_time_target)
    list.append(track_time_output)
    return list

    # #ignore = abs(track_time_target - track_time_output)
    # for index in range(19):
    #     #plt.plot(ignore[index], 'r--',label='Target Index: ' + str(index))
    #     plt.plot(track_time_target[index], 'r--',label='Target Index: ' + str(index))
    #     plt.plot(track_time_output[index], 'b-',label='Output Index: ' + str(index))
    #     plt.legend( loc='upper right' )
    #     plt.savefig('Results/' + 'Index' + str(index) + '.png')
    #     plt.show()

def mutate(model_in, model_out, many_strength = 1, much_strength = 1):
    #NOTE: Takes in_num file, mutates it and saves as out_num file, many_strength denotes how many mutation while
    # much_strength controls how strong each mutation is

    w = model_in.get_weights()
    for many in range(many_strength):#Number of mutations
        i = randint(0, len(w)-1)
        if len(w[i].shape) == 1: #Bias
            j = randint(0, len(w[i])-1)
            w[i][j] += np.random.normal(-0.1 * much_strength, 0.1 * much_strength)
            # if (randint(1, 100) == 5): #SUPER MUTATE
            #     w[i][j] += np.random.normal(-1 * much_strength, 1 * much_strength)
        else:  # Bias
            j = randint(0, len(w[i]) - 1)
            k = randint(0, len(w[i][j]) - 1)
            w[i][j][k] += np.random.normal(-0.1 * much_strength, 0.1 * much_strength)
            # if (randint(1, 100) == 5):  # SUPER MUTATE
            #     w[i][j][k] += np.random.normal(-1 * much_strength, 1 * much_strength)

    model_out.set_weights(w) #Save weights





def rec_weakness(train_data, model, n_prev=7, novelty = False, test = False): #Calculates weakness (anti fitness) of RECCURRENT models
    weakness = np.zeros(19)
    input = np.reshape(train_data[0:n_prev], (1, n_prev, 21))  #First training example in its entirety

    for example in range(len(train_data)-n_prev):#For all training examples
        model_out = model.predict(input) #Time domain simulation
        #Calculate error (weakness)
        for index in range(19):
            weakness[index] += math.fabs(model_out[0][index] - train_data[example+n_prev][index])#Time variant simulation
        #Fill in new input data
        for k in range(len(model_out)): #Modify the last slot
            input[0][0][k] = model_out[0][k]
            input[0][0][k] = model_out[0][k]
        #Fill in two control variables
        input[0][0][19] = train_data[example+n_prev][19]
        input[0][0][20] = train_data[example+n_prev][20]
        input = np.roll(input, -1, axis=1)  # Track back everything one step and move last one to the last row
    if (novelty):
        return weakness
    elif (test == True):
        return np.sum(weakness)/(len(train_data)-n_prev)
    else:
        return np.sum(np.square(weakness))

def ff_weakness(train_data, model, novelty = False, test = False): #Calculates weakness (anti fitness) of FEED-FORWARD models
    weakness = np.zeros(19)
    input = np.reshape(train_data[0], (1, 21)) #First training example in its entirety
    for example in range(len(train_data)-1):#For all training examples
        model_out = model.predict(input) #Time domain simulation
        #timeless_sim = model.predict(np.reshape(train_data[example], (1, 21)))
        #Calculate error (weakness)
        for index in range(19):
            weakness[index] += math.fabs(model_out[0][index] - train_data[example+1][index])#Time variant simulation
            #weakness += pow(math.fabs(timeless_sim[0][index] - train_data[example+1][index]), 2)#Add non-timed simulation signal
        #Fill in new input data
        for k in range(len(model_out)):
            input[0][k] = model_out[0][k]
            input[0][k] = model_out[0][k]
        #Fill in two control variables
        input[0][19] = train_data[example+1][19]
        input[0][20] = train_data[example+1][20]
    if (novelty):
        return weakness
    elif (test == True):
        return np.sum(weakness)/(len(train_data)-1)
    else:
        return np.sum(np.square(weakness))