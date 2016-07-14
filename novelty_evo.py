import numpy as np, re, theanets
from random import randint
import modules_simulator as mod
import bottleneck
from keras.models import model_from_json



#MACRO
import_new = True   #Set to true if import new seed from Model/Singular
new_popn = True #Set to true if initializing a new population from Evolutionary/0
seed_size = 100 #Size of initial seed population
obj_quota = 10 #Obj_quota are the number of population selected by traditional objective based fitness
infeasible = 5 #Quantity of popn selected despite being infeasible
n_prev = 7
archive_size_limit = 1500 #Limits the max size of the archive
total_gen = 25000
archiving_rate = 2 #Num of best novelties added to the archive every generation
is_recurrent = False



#Method to save/copy models
def import_models(in_filename, save_filename, model_arch):
    model_arch.load_weights(in_filename)
    model_arch.save_weights(save_filename, overwrite=True)
    model_arch.compile(loss='mse', optimizer='rmsprop')

def save_models(nn_popn, popn_size):
    for i in range(popn_size):
        nn_popn[i].save_weights('Evolutionary/' + str(i), overwrite=True)

def get_model_arch(seed = 'Evolutionary/seed.json'): #Get model architecture
    import json
    with open(seed) as json_file:
        json_data = json.load(json_file)
    model_arch = model_from_json(json_data)
    return model_arch



def dataio(split = 1000):
    data, max, min = mod.data_preprocess()#Process data and populate global variable train_data
    train_data = data[0:split]
    valid_data = data[split:len(data)]
    return train_data, valid_data

def main():

    constraint_coefficient = 5 #Determines the search space of feasibility relative to the best weakness
    popn_size = 2 * seed_size
    ##### DATA IO ######
    train_data, valid_data = dataio()
    model_arch = get_model_arch() #Get model architecture
    model_arch.load_weights('Evolutionary/seed.h5')


    ### DECLARE A 3-tuple LIST TO HOLD POPULATION, FITNESS & NOVELTY respectively
    population = np.zeros(popn_size * 3, dtype=np.float64)
    population = np.reshape(population, (popn_size, 3))
    for x in range(popn_size): # Naming our models
        population[x][0] = x

    # Form the neural network correspnding to population (nn_popn)
    if (new_popn):  # Form new population
        if (import_new):  # Import model
            import_models('Evolutionary/seed.h5', 'Evolutionary/0', model_arch)

        nn_popn = []
        for i in range(popn_size):
            nn_popn.append(get_model_arch())
            nn_popn[i].load_weights('Evolutionary/0')
            nn_popn[i].compile(loss='mae', optimizer='adam')

        for i in range(popn_size - 1):
            mod.mutate(nn_popn[0], nn_popn[i + 1], 10, 10)

        #Invoke generation, archive and metrics
        track_valid = np.zeros(1)#Metrics tracker
        weakness_tracker = np.zeros(1)#MSE tracker
        generation = 0
        archive = np.empty([1,19])
        if is_recurrent:
            seed_archive = mod.rec_weakness(train_data, nn_popn[0], n_prev, True)
        else:
            seed_archive = mod.ff_weakness(train_data, nn_popn[0], True)
        for index in range (19):
            archive[0][index] = seed_archive[index]

    else:

        nn_popn = []
        for i in range(popn_size):
            nn_popn.append(model_arch)
            nn_popn[i].load_weights('Evolutionary/' + str(i))
            model_arch.compile(loss='mae', optimizer='adam')

        #Load archive, generation and metrics
        track_valid = np.loadtxt('Evolutionary/Files/valid.csv', delimiter=',' )
        weakness_tracker = np.loadtxt('Evolutionary/Files/weakness.csv', delimiter=',' )

        archive = np.loadtxt('Evolutionary/Files/archive.csv', delimiter=',' )
        generation = int(np.loadtxt('Evolutionary/Files/generation.csv', delimiter=',' ))


    ################# LOOP ######################
    best_weakness = 1000000000 #Magic large constraint to start with (IGNORE)
    gen_since = 0 #Generation since last progress
    constraint = 10000000 #Magic large constraint to start with (IGNORE)

    #Create a list to hold weakness_matrix to reduce weakness calculations
    weakness_list = []

    gen_array = np.empty(1)
    restart_assess = True
    if (new_popn or import_new):
        restart_assess = False


    while generation < total_gen:
        generation = generation + 1

        #NATURAL SELECTION
        for x in range(popn_size): #Evaluate weakness and novelty for each
            if (generation == 1 or restart_assess): #For first time
                if is_recurrent:
                    weak_matrix = mod.rec_weakness(train_data, nn_popn[int(population[x][0])], n_prev, novelty=True)
                else:
                    weak_matrix = mod.ff_weakness(train_data, nn_popn[int(population[x][0])], novelty=True)
                weakness_list.append(weak_matrix)
                population[x][1] = np.sum(np.square(weak_matrix))#Assign weakness
            elif(x < seed_size):
                weak_matrix = weakness_list[int(population[x][0])]
            else:
                if is_recurrent:
                    weak_matrix = mod.rec_weakness(train_data, nn_popn[int(population[x][0])], n_prev, novelty=True)
                else:
                    weak_matrix = mod.ff_weakness(train_data, nn_popn[int(population[x][0])], novelty=True)
                weakness_list[int(population[x][0])] = weak_matrix
                population[x][1] = np.sum(np.square(weak_matrix))#Assign weakness

            #Assign novelty and implement constraints (feasibility)
            if (population[x][1] > constraint):
                population[x][2] = 0 #MCNS step where infeasible popn is given 0.01 novelty score
            else:
                population[x][2] = mod.novelty(weak_matrix, archive)#Assign novelty metric
        restart_assess = False
        population = population[population[:,2].argsort()][::-1]##Ranked on novelty s.t. 0 index is the best (highest feasible novelty)

        #Objective based of population selection
        sort_start = seed_size - obj_quota - infeasible
        obj_set = bottleneck.argpartsort(population[sort_start:,1], obj_quota)[:obj_quota]

        #Make the changes
        for x in range(obj_quota):
            selectee = obj_set[x] + sort_start
            population[sort_start + x][0], population[selectee][0]  = population[selectee][0], population[sort_start + x][0]
            population[sort_start + x][2], population[selectee][2]  = population[selectee][2], population[sort_start + x][2]
            population[sort_start + x][1], population[selectee][1]  = population[selectee][1], population[sort_start + x][1]


        #Infeasible portion of population selection
        sort_start = seed_size - infeasible
        infeasible_set = -bottleneck.argpartsort(-population[sort_start:,1], infeasible)[:infeasible]

        #Make the changes
        for x in range(infeasible):
            selectee = infeasible_set[x] + sort_start
            population[sort_start + x][0], population[selectee][0]  = population[selectee][0], population[sort_start + x][0]
            population[sort_start + x][2], population[selectee][2]  = population[selectee][2], population[sort_start + x][2]
            population[sort_start + x][1], population[selectee][1]  = population[selectee][1], population[sort_start + x][1]

        #Mutate to renew population
        for x in range(seed_size):
            many = 1
            much = randint(1,5)
            if (randint(1,100) == 91):
                many = randint(1,10)
                much = randint(1,100)
            mod.mutate(nn_popn[int(population[x][0])], nn_popn[int(population[x+seed_size][0])], many, much)

        ########### End of core evol. algorithm #######

        #Update archive
        for l in range (archiving_rate):
            weak_matrix = np.reshape(weakness_list[int(population[l][0])], (1,19))
            archive = np.append(archive, weak_matrix, axis=0)

            #Limit archive size
            cut = len(archive) - archive_size_limit
            if (cut > 0):
                for i in range(cut):
                    archive = np.delete(archive, 0, 0)

        #Save archive, generation and Metrics periodically
        if (generation % 500 == 0):
            #METRICS
            best_weakness_index = np.argmin(population[:,1])


            #Generation
            gen_array[0] = generation
            np.savetxt('Evolutionary/Files/generation.csv', gen_array, delimiter=',')

            #Archive
            np.savetxt('Evolutionary/Files/archive.csv', archive, delimiter=',')


         ##UI
        if (np.amin(population[:,1])/len(train_data) < best_weakness):
            gen_since = 0
            best_weakness = np.amin(population[:,1])/len(train_data)
        else:
            gen_since += 1

        #Method to save models and track weakness
        if ((generation) % 100 == 0):
            save_models(nn_popn, popn_size)

            if (generation == 1):
                weakness_tracker = np.delete(weakness_tracker, 0)

        if (generation % 10 == 0):
            if is_recurrent:
                valid_score = mod.rec_weakness(valid_data, nn_popn[int(population[np.argmin(population[:,1])][0])], n_prev)/ len(valid_data)
            else:
                valid_score = mod.ff_weakness(valid_data, nn_popn[int(population[np.argmin(population[:, 1])][0])]) / len(valid_data)
            print ('Gen:', generation, ' Best:','%.2f' % best_weakness, 'Valid: ', '%.2f' % valid_score,'Failed:', gen_since, )

            weakness_tracker = np.append(weakness_tracker, best_weakness)
            np.savetxt('Evolutionary/Files/weakness.csv', weakness_tracker, delimiter=',')
            track_valid = np.append(track_valid, valid_score)
            np.savetxt('Evolutionary/Files/valid.csv', track_valid, delimiter=',')

        #Update constraints
        constraint = best_weakness * constraint_coefficient
        if (generation % 1500 == 0 and constraint_coefficient > 2):
            constraint_coefficient = constraint_coefficient - 0.5





if __name__ == '__main__':
    main()
