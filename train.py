import numpy as np, keras, sys
from matplotlib import pyplot as plt
plt.switch_backend('Qt4Agg')
import modules_simulator as mod
import theanets

#Macro Variables
filename = "Models/singular"
num_hnode = 35
stat_run = 10000

def save_model(model):
    import json
    json_string = model.to_json()
    with open('Evolutionary/seed.json', 'w') as outfile:
        json.dump(json_string, outfile)
    model.save_weights('Evolutionary/seed.h5', overwrite=True)



def main():

    train_data, max, min = mod.data_preprocess()

    #Shuffle data
    #shuffled = np.random.permutation(train_data)
    shuffled = train_data#no shuffle
    train_size = len(train_data)

    #Setup training data
    delete = np.split(shuffled, [train_size - 1, train_size])
    train_X = delete[0]

    #Setup train_Y
    ignore = np.split(shuffled, [1, train_size])
    ignore1 = np.array(ignore[1])
    ignore2 = np.hsplit(ignore1, [19, 21])
    train_Y = ignore2[0]

    #Setup Validation
    ig = np.split(train_X, [1000, 1224])
    ig2 = np.split(train_Y, [1000, 1224])
    train_X = ig[0]
    train_Y = ig2[0]

    valid_X = ig[1]
    valid_Y = ig2[1]

    print(train_X.shape, train_Y.shape, valid_X.shape, valid_Y.shape)
    #sys.exit()


    ##################### END OF DATA IO ##########################
    best = 100000000
    for st in range (stat_run):
        #Create the simulator
        model = mod.init_model(num_hnode)
        model.fit(train_X, train_Y, nb_epoch=15, batch_size=32, verbose=0, shuffle=True, validation_data=(valid_X, valid_Y))
        weak = mod.ff_weakness(train_X, model)
        if weak < best:
            best = weak
            print (weak/len(train_X))
            save_model(model)























if __name__ == '__main__':
    main()
