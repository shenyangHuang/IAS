from MLP_Controller import Controller
from Dataset_Management import mlp_datasets
from MLP_archSearch import MLP_SAS 
import numpy as np
import os.path
import pickle
import datetime


'''
randomize the order of which class arrives
in the paper, a default order from 0-9 is used 
'''
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
    epoch_limit = 20
    N_workers = 8
    restart_num = 1
    default = True

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
    set up training, validation and testing data
    '''
    print("training step 0")
    datasets = mlp_datasets.Incremental_MNISTlike("MNIST", 0.1, order,normalization="numerical")
    datasets = mlp_datasets.sum_data(datasets,1,0)

    '''
    Training initial teacher network 
    '''
    accuracy = main.train_Teacher(datasets[0], architecture=[16],
                                    input_dim=784, output_dim=2, 
                                    lr=lr,epochs=epochs, batch_size=32, 
                                    random_seed=seed,mute=True,test=True,class_curve=True)
    print("Teacher network achieved test accuracy " + str(accuracy[1]))
    params = main.export()

    '''
    SAS_Controller manages neural architecture internally
    details see MLP_Controller/Controller.py
    SAS stands for simple architecture search (it is the neural architecture search principle used here)
    '''
    SAS_Controller = Controller.Controller(seed,'sas')
    SAS_Controller.param_import(params)

    '''
    Starting Incremental Architecture Search procedure
    '''
    for i in range(2,10):
        print ("--------------------------------------------------------------")
        print("training step " + str(i-1))
        print ("saturate with new data")

        '''
        concatenate new data
        '''
        datasets = mlp_datasets.sum_data(datasets,i,0)
        SASpath = "./logs/"+str(i)+"_1SAS"

        '''
        Train with concatenated dataset using early stopping
        '''
        SAS_accuracy = SAS_Controller.execute(datasets[0],input_dim=784,
                                            output_dim=(i+1),lr=lr,epochs=epochs,
                                            batch_size=32,random_seed=seed,mute=True,
                                            test=True,noVal=False,log_path=SASpath,class_curve=True)

        print ("before SAS search, model achieved test accuracy of " + str(SAS_accuracy[1]))
        SAS_Controller.class_accuracy(datasets[0],input_dim=784,output_dim=(i+1),lr=lr,batch_size=32,mute=True)

        '''
        Retrain a baseline from scratch using the same architecture
        '''
        base_accuracy = SAS_Controller.baseline(datasets[0],input_dim=784,output_dim=(i+1),lr=lr,epochs=epochs,batch_size=32,random_seed=seed,mute=True,
                                                            test=True,noVal=False,upload=False,class_curve=True)
        print("Baseline achieved test accuracy " + str(base_accuracy[1]))
        SAS_params = SAS_Controller.export()
        
        '''
        Simple Architecture Search using 6 predefined guide lines 
        For details see MLP_archSearch/MLP_SAS.py
        obtained the best architecture for this step 
        '''
        SAS_best = MLP_SAS.simple_search(datasets[0],SAS_params,
                                        SAS_accuracy[1],N_workers,
                                        restart_num=restart_num,output_dim=(i+1),
                                        lr=lr,epoch_limit=epoch_limit,random_seed=seed)

        print("obtained SAS candidate")
        print ("--------------------------------------------------------------")
        print("Doing Further Training")

        '''
        ensure that the training is complete for the current dataset 
        usually runs very few epoches
        '''
        SAS_Controller.param_import(SAS_best['params'])
        SAS_test = SAS_Controller.test(datasets[0],input_dim=784,output_dim=(i+1),lr=lr,batch_size=32,mute=True)
        print("SAS candidate achieved validation acccuracy of " + str(SAS_best['accuracy']) + " before further training ")
        print("SAS candidate achieved test accuracy of " + str(SAS_test[1]) + " before further training ")
        SASpath = "./logs/"+str(i)+"_2SAS"

        SAS_accuracy = SAS_Controller.execute(datasets[0],input_dim=784,
                                            output_dim=(i+1),lr=lr,epochs=epochs,
                                            batch_size=32,random_seed=seed,mute=True,
                                            test=True,noVal=False,log_path=SASpath,class_curve=True)

        '''
        revert to pretraining weights if accuracy doesn't improve
        '''
        print("SAS candidate achieved test accuracy " + str(SAS_accuracy[1]) + " after further training")
        if(SAS_accuracy[1] <= SAS_test[1]):
            SAS_Controller.param_import(SAS_best['params'])
            print("revert to pretrained weights with test accuracy of " + str(SAS_test[1]) + " before further training")
        SAS_Controller.class_accuracy(datasets[0],input_dim=784,output_dim=(i+1),lr=lr,batch_size=32,mute=True)
        
        '''
        Retrain a baseline from scratch using the same architecture
        This baseline is reported in the graph of the paper 
        '''
        base_accuracy = SAS_Controller.baseline(datasets[0],input_dim=784,
                                                output_dim=(i+1),lr=lr,
                                                epochs=epochs,batch_size=32,
                                                random_seed=seed,mute=True,
                                                test=True,noVal=False,upload=False,class_curve=True)

        print("Baseline achieved test accuracy " + str(base_accuracy[1]) + " after further training")
        SAS_params = SAS_Controller.export()
        print("storing best result from time step " +str(i))
        print(datetime.datetime.now())

        '''
        store the best model from this step 
        '''
        with open(str(i)+'.pkl', 'wb') as f:
            pickle.dump(SAS_Controller, f, pickle.HIGHEST_PROTOCOL)
        print('')
        print('')

if __name__ == "__main__":
   main()



