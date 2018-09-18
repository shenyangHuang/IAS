from MLP_Controller import Controller
from Dataset_Management import mlp_datasets
from MLP_archSearch import MLP_SAS 
import numpy as np
import os.path
import pickle
import datetime

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
    set up training, validation and testing data
    '''
    datasets = mlp_datasets.All_MNISTlike(0.1, normalization="numerical")
    #use 10 MNIST classes as base knowledge
    for i in range(1,10):
        mnist_datasets = mlp_datasets.sum_data(datasets,i,0)

    main =  Controller.Controller(seed,'main')

    print("training step 0")
    accuracy = main.train_Teacher(datasets[0], architecture=[256,256],
                                    input_dim=784, output_dim=10, 
                                    lr=lr,epochs=epochs, batch_size=32, 
                                    random_seed=seed,mute=True,test=True,class_curve=True)
    print("Teacher network achieved test accuracy " + str(accuracy[1]))
    params = main.export()
    SAS_Controller = Controller.Controller(seed,'sas')
    SAS_Controller.param_import(params)

    '''
    Starting Incremental Architecture Search procedure
    '''
    for i in range(10,20):
        print ("--------------------------------------------------------------")
        print("training step " + str(i-9))
        print ("saturate with new data")
        SASpath = "./logs/"+str(i-9)+"_1SAS"

        SAS_accuracy = SAS_Controller.execute(datasets[0],input_dim=784,
                                            output_dim=(i+1),lr=lr,epochs=epochs,
                                            batch_size=32,random_seed=seed,mute=True,
                                            test=True,noVal=False,log_path=SASpath,class_curve=True)

        print ("before SAS search, model achieved test accuracy of " + str(SAS_accuracy[1]))
        SAS_Controller.class_accuracy(datasets[0],input_dim=784,output_dim=(i+1),lr=lr,batch_size=32,mute=True)

        base_accuracy = SAS_Controller.baseline(datasets[0],input_dim=784,output_dim=(i+1),lr=lr,epochs=epochs,batch_size=32,random_seed=seed,mute=True,
                                                            test=True,noVal=False,upload=False,class_curve=True)
        print("Baseline achieved test accuracy " + str(base_accuracy[1]))
        SAS_params = SAS_Controller.export()
        
        SAS_best = MLP_SAS.simple_search(datasets[0],SAS_params,
                                        SAS_accuracy[1],N_workers,
                                        restart_num=restart_num,output_dim=(i+1),
                                        lr=lr,epoch_limit=epoch_limit,random_seed=seed)

        print("obtained SAS candidate")
        print ("--------------------------------------------------------------")
        print("Doing Further Training")

        SAS_Controller.param_import(SAS_best['params'])
        SAS_test = SAS_Controller.test(datasets[0],input_dim=784,output_dim=(i+1),lr=lr,batch_size=32,mute=True)
        print("SAS candidate achieved validation acccuracy of " + str(SAS_best['accuracy']) + " before further training ")
        print("SAS candidate achieved test accuracy of " + str(SAS_test[1]) + " before further training ")
        SASpath = "./logs/"+str(i)+"_2SAS"

        SAS_accuracy = SAS_Controller.execute(datasets[0],input_dim=784,
                                            output_dim=(i+1),lr=lr,epochs=epochs,
                                            batch_size=32,random_seed=seed,mute=True,
                                            test=True,noVal=False,log_path=SASpath,class_curve=True)

        #revert to pretrained weights if accuracy doesn't improve
        print("SAS candidate achieved test accuracy " + str(SAS_accuracy[1]) + " after further training")
        if(SAS_accuracy[1] <= SAS_test[1]):
            SAS_Controller.param_import(SAS_best['params'])
            print("revert to pretrained weights with test accuracy of " + str(SAS_test[1]) + " before further training")
        SAS_Controller.class_accuracy(datasets[0],input_dim=784,output_dim=(i+1),lr=lr,batch_size=32,mute=True)
        
        base_accuracy = SAS_Controller.baseline(datasets[0],input_dim=784,
                                                output_dim=(i+1),lr=lr,
                                                epochs=epochs,batch_size=32,
                                                random_seed=seed,mute=True,
                                                test=True,noVal=False,upload=False,class_curve=True)

        print("Baseline achieved test accuracy " + str(base_accuracy[1]) + " after further training")
        SAS_params = SAS_Controller.export()
        print("storing best result from time step " +str(i))
        print(datetime.datetime.now())
        with open(str(i)+'.pkl', 'wb') as f:
            pickle.dump(SAS_Controller, f, pickle.HIGHEST_PROTOCOL)
        print('')
        print('')


    #train with all data in the end
    SAS_accuracy = SAS_Controller.execute(datasets[0],input_dim=784,
                                        output_dim=(i+1),lr=lr,epochs=epochs,
                                        batch_size=32,random_seed=seed,mute=True,
                                        test=True,noVal=True,log_path="none",class_curve=True)
    print("SAS achieved test accuracy " + str(SAS_accuracy[1]) + " with all MNIST + Fashion-MNIST data")

    base_accuracy = SAS_Controller.baseline(datasets[0],input_dim=784,
                                            output_dim=(i+1),lr=lr,
                                            epochs=epochs,batch_size=32,
                                            random_seed=seed,mute=True,
                                            test=True,noVal=True,upload=False,class_curve=True)
    print("Baseline achieved test accuracy " + str(base_accuracy[1]) + " with all MNIST + Fashion-MNIST data")


if __name__ == "__main__":
   main()






