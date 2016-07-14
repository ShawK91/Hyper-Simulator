import numpy as np, sys, copy
from random import randint
import modules_simulator as mod
from keras.models import model_from_json

#MACRO
import_new = True   #Set to true if import new seed from Model/Singular
new_popn = True #Set to true if initializing a new population from Evolutionary/0
seed_size = 100 #Size of initial seed population
freeloaders = 5 #Freeloaders are weak candidates that are kept in popn by pure luck (freeloaders = 0 --> elitism)
total_gen = 25000
is_recurrent = False
n_prev = 7



#Method to save/copy models
def import_models(in_filename, save_filename, model_arch):
    model_arch.load_weights(in_filename)
    model_arch.save_weights(save_filename, overwrite=True)
    model_arch.compile(loss='mae', optimizer='adam')

def save_models(nn_popn, popn_size):
    for i in range(popn_size):
        nn_popn[i].save_weights('Evolutionary/' + str(i), overwrite=True)

def get_model_arch(seed = 'Evolutionary/seed.json'): #Get model architecture
    import json
    with open(seed) as json_file:
        json_data = json.load(json_file)
    model_arch = model_from_json(json_data)
    return model_arch

def check_weakness(foldername, iterate = 200):

	train_data, _,_ = mod.data_preprocess()
	best_weakness = 100000
	for i in range(200):
		model = get_model_arch()
		model.load_weights('Evolutionary/' + str(i))
		weakness = mod.ff_weakness(train_data, model, test=True)
		if weakness < best_weakness:
			best_weakness = weakness
			print i, best_weakness



def dataio(split = 1000):
    data, max, min = mod.data_preprocess()#Process data and populate global variable train_data
    train_data = data[0:split]
    valid_data = data[split:len(data)]
    return train_data, valid_data

def main():

    mod.print_results('sim.h5')
    #check_weakness('Evolutionary/')


    # model = get_model_arch()
    # model.load_weights('Evolutionary/seed.h5')
    # train_data, valid_data = dataio()
    # weak = mod.ff_weakness(train_data, model)
    # print (weak / len(train_data))
    #



    

if __name__ == '__main__':
    main()
