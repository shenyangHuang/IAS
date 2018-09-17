import datetime
from MLP_Controller import Controller
from pathos.pools import ParallelPool
import numpy as np
import copy




def Just_Train(dataset,output_dim,params,instruction,lr,epoch_limit,restart_num,random_seed=29):    
    group_nets = []
    max_accuracy = 0
    output_params = 0
    architecture = params['architecture']
    WiderInstructions = instruction['Wider']
    DeeperInstructions = instruction['Deeper']

    for i in range(0,restart_num):
        net = Controller.Controller(i,str(i))
        net.param_import(params)
        if(len(WiderInstructions) > 0):
            for i in range(len(WiderInstructions)):
                neurons = architecture[WiderInstructions[i]-1]
                net.Net2WiderNet(WiderInstructions[i],neurons*2,0.1)
        if(len(DeeperInstructions) > 0):
            for i in range(len(DeeperInstructions)):
                net.Net2DeeperNet(DeeperInstructions[i])
        group_nets.append(net)

    for i in range(0,len(group_nets)):
        output = group_nets[i].execute(dataset,input_dim=784,output_dim=output_dim,lr=lr,epochs=epoch_limit,batch_size=32,random_seed=random_seed,mute=True,
                test=False)
        accuracy = output[1]
        if(accuracy > max_accuracy):
            max_accuracy = accuracy
            output_params = group_nets[i].export()
    return {'accuracy':max_accuracy,'params':output_params}
    

def simple_search(dataset,params,val_accuracy,N_workers,restart_num=3,output_dim=10,lr=0.001,epoch_limit=20,random_seed=29):

    print ("--------------------------------------------------------------")
    print("SAS starts")
    print ("--------------------------------------------------------------")
    print("run for at most " + str(epoch_limit) + " epochs each")
    print("running in parallel using " + str(N_workers) + " workers ")
    print(datetime.datetime.now())

    val_param = copy.deepcopy(params)
    architecture = params['architecture']
    layer = len(architecture)
    instructions = []
    instructions.append({'Wider':[layer],'Deeper':[]})
    instructions.append({'Wider':[layer,layer],'Deeper':[]})
    instructions.append({'Wider':[],'Deeper':[layer]})
    instructions.append({'Wider':[],'Deeper':[layer,layer+1]})
    instructions.append({'Wider':[layer],'Deeper':[layer]})
    instructions.append({'Wider':[],'Deeper':[layer]})
    instructions.append({'Wider':[layer,layer],'Deeper':[layer]})
    EveryLayer = np.arange(layer) + 1
    #Widen all layers at once 
    instructions.append({'Wider':EveryLayer,'Deeper':[]})

    print("instruction generation complete")
    pool = ParallelPool(N_workers)
    print(" creating Pool and setting up workers")
    num_instructions = len(instructions)

    l_dataset = [dataset]*num_instructions
    l_output_dim = [output_dim]*num_instructions
    l_params = [params]*num_instructions
    l_lr = [lr]*num_instructions
    l_epoch_limit = [epoch_limit]*num_instructions
    l_restart_num = [restart_num]*num_instructions
    l_seed = [random_seed]*num_instructions


    print("function call preparation complete ")
    candidates = pool.map(Just_Train,l_dataset,l_output_dim,l_params,instructions,l_lr,l_epoch_limit,l_restart_num,l_seed)
    print("all candidates received")
    print(datetime.datetime.now())

    best_accu = 0
    best_param = 0
    # find the best candidate
    for candidate in candidates:
        if candidate is None:
            print("find a none type")
            continue
        accuracy = candidate['accuracy']
        architecture = candidate['params']['architecture']
        print ("for architecture : " + str(architecture))
        print (" achieved validation accuracy of " + str(accuracy))
        if(accuracy > best_accu):
            best_accu = accuracy
            best_param = candidate['params']

    if(val_accuracy > best_accu):
        best_accu = val_accuracy
        best_param = val_param

    print("best candidate has architecture")
    print(best_param['architecture'])

    return {'accuracy':best_accu,'params':best_param}
