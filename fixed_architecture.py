from MLP_Controller import Controller
from Dataset_Management import mlp_datasets
import numpy as np
import os.path
import pickle
import datetime

def setup_Fashion(ratio=0.1):
    datasets = mlp_datasets.Incremental_MNISTlike("Fashion", 0.1, normalization="numerical")
    return datasets

def random_order(default=False):
    arr = np.arange(10)
    if(not default):
        np.random.shuffle(arr)
    return (arr)


def main():
    '''
    hyperparameters 
    '''
    lr=0.0001
    seed = 29
    epochs = 200
    default = True
    percentage = 0.1 


    '''
    determine the arrival order of classes
    '''
    file_name = "order.pickle"
    order=[]
    if(os.path.isfile(file_name)):
        with open(file_name, 'rb') as f:
            order = pickle.load(f)
    else:
        order = random_order(default=default)
        with open(file_name, 'wb') as f:
            pickle.dump(order, f, pickle.HIGHEST_PROTOCOL)
            f.close()
    print ("The ordering of classes is :")
    print (order)
    main =  Controller.Controller(seed,'main')


    '''
    Training initial teacher network 
    '''
    print("training step 0")
    datasets = mlp_datasets.Incremental_MNISTlike("MNIST", 0.1, order,normalization="numerical")
    datasets = mlp_datasets.sum_data(datasets,1,0)

    accuracy = main.train_Teacher(datasets[0], architecture=[256,256],
                                    input_dim=784, output_dim=2, 
                                    lr=lr,epochs=epochs, batch_size=32, 
                                    random_seed=seed,mute=True,test=True,class_curve=True)
    print("Teacher network achieved test accuracy " + str(accuracy[1]))

    '''
    Starting Incremental Architecture Search procedure
    '''
    for i in range(2,10):
        print ("--------------------------------------------------------------")
        print("training step " + str(i-1))
        print ("saturate with new data")
        datasets = mlp_datasets.sum_data(datasets,i,0)
        fixedpath = "./logs/"+str(i)+"_1fixed"
        fixed_accuracy = main.execute(datasets[0],input_dim=784,
                                            output_dim=(i+1),lr=lr,epochs=epochs,
                                            batch_size=32,random_seed=seed,mute=True,
                                            test=True,noVal=False,log_path=fixedpath,class_curve=True)

        print("fixed architecture achieved test accuracy " + str(fixed_accuracy[1]) + "at step " + str(i-1))

        print("storing best result from time step " +str(i))
        print(datetime.datetime.now())
        with open(str(i)+'.pkl', 'wb') as f:
            pickle.dump(main, f, pickle.HIGHEST_PROTOCOL)
        print('')
        print('')

    #train with all MNIST data in the end
    fixed_accuracy = main.execute(datasets[0],input_dim=784,
                                        output_dim=(i+1),lr=lr,epochs=epochs,
                                        batch_size=32,random_seed=seed,mute=True,
                                        test=True,noVal=True,log_path="none",class_curve=True)
    print("fixed architecture achieved test accuracy " + str(fixed_accuracy[1]) + " with all MNIST data")


if __name__ == "__main__":
   main()



