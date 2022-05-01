import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse,time
import math
from copy import deepcopy
import logging
from layers import Conv2d, Linear

eplison_1 = 0.2
eplison_2 = 0.01
#  Define MLP model
class MLPNet(nn.Module):
    def __init__(self, n_hidden=100, n_outputs=10):
        super(MLPNet, self).__init__()
        self.act=OrderedDict()
        self.lin1 = Linear(784,n_hidden,bias=False)
        self.lin2 = Linear(n_hidden,n_hidden, bias=False)
        self.fc1  = Linear(n_hidden, n_outputs, bias=False)
        

    def forward(self, x, space1= [None, None, None], space2= [None, None, None]):
        # regime 2:
        if space1[0] is not None or space2[0] is not None:
            self.act['Lin1']=x
            x = self.lin1(x, space1=space1[0], space2 = space2[0])        
            x = F.relu(x)
            self.act['Lin2']=x
            x = self.lin2(x, space1=space1[1], space2 = space2[1])        
            x = F.relu(x)
            self.act['fc1']=x
            x = self.fc1(x,space1=space1[2], space2 = space2[2])
        else:
            self.act['Lin1']=x
            x = self.lin1(x)        
            x = F.relu(x)
            self.act['Lin2']=x
            x = self.lin2(x)        
            x = F.relu(x)
            self.act['fc1']=x
            x = self.fc1(x)           
        return x 


def get_model(model):
    return deepcopy(model.state_dict())

def set_model(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def save_model(model, memory, savename):
    ckpt = {
        'model': model.state_dict(),
        'memory': memory,
    }

    # Save to file.
    torch.save(ckpt, savename+'checkpoint.pt')
    print(savename)

    return 

def train (args, model, device, x, y, optimizer,criterion):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b].view(-1,28*28)
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output, target)        
        loss.backward()
        optimizer.step()

def train_projected (args, model,device,x,y,optimizer,criterion,feature_mat):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b].view(-1,28*28)
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output, target)         
        loss.backward()        
        # Gradient Projections 
        for k, (m,params) in enumerate(model.named_parameters()):
            sz =  params.grad.data.size(0)
            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[k]).view(params.size())
        optimizer.step()


def train_projected_regime (args, model,device,x,y,optimizer,criterion,memory, task_name, task_name_list, task_id, feature_mat, space1=[None, None, None], space2=[None, None, None]):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b].view(-1,28*28)
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data, space1=space1, space2=space2)
        loss = criterion(output, target)  

        loss.backward()        

        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if 'weight' in m:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[kk]).view(params.size())
                kk+=1 


        optimizer.step()
    
def test (args, model, device, x, y, criterion, space1=[None, None, None], space2=[None, None, None]):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b].view(-1,28*28)
            data, target = data.to(device), y[b].to(device)
            output = model(data, space1=space1, space2=space2)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def get_representation_and_gradient(net, device, optimizer, criterion, task_id, x, y=None):
    # Collect activations by forward pass
    # Collect gradient by backward pass
    # net.eval()
    steps = 1
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:300] # Take random training samples
    example_data = x[b].view(-1,28*28)
    example_data, target = example_data.to(device), y[b].to(device)
    
    batch_list=[300,300,300] 
    mat_list=[] # list contains representation matrix of each layer
    grad_list=[] # list contains gradient of each layer
    act_key=list(net.act.keys())
    # example_out  = net(example_data)

    net.eval()
    example_out  = net(example_data)

    for k in range(len(act_key)):
        bsz=batch_list[k]
        act = net.act[act_key[k]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)
    
    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list, grad_list

def get_space_and_grad(model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all):
    print ('Threshold: ', threshold) 
    Ours = True
    if task_name == 'pmnist-0':
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]

            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  

            # save into memory
            memory[task_name][str(i)]['space_list'] = U[:,0:r]

            space_list_all.append(U[:,0:r])


    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]

            if Ours:
                #=1. calculate the projection using previous space
                print_log('activation shape:{}'.format(activation.shape), log)
                print_log('space shape:{}'.format(space_list_all[i].shape), log)
                delta = []
                R2 = np.dot(activation,activation.transpose())
                for ki in range(space_list_all[i].shape[1]):
                    space = space_list_all[i].transpose()[ki]
                    # print(space.shape)
                    delta_i = np.dot(np.dot(space.transpose(), R2), space)
                    # print(delta_i)
                    delta.append(delta_i)
                delta = np.array(delta)
                
                #=2  following the GPM to get the sigma (S**2)
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                
                act_hat = activation
                act_hat -= np.dot(np.dot(space_list_all[i],space_list_all[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                sigma = S**2


                #=3 stack delta and sigma in a same list, then sort in descending order
                stack = np.hstack((delta, sigma))  #[0,..30, 31..99]
                stack_index = np.argsort(stack)[::-1]   #[99, 0, 4,7...]
                #print('stack index:{}'.format(stack_index))
                stack = np.sort(stack)[::-1]
                
                #=4 select the most import basis
                r_pre = len(delta)
                r = 0
                accumulated_sval = 0
                for ii in range(len(stack)):
                    if accumulated_sval < threshold[i] * sval_total:
                        accumulated_sval += stack[ii]
                        r += 1
                        if r == activation.shape[0]:
                            break
                    else:
                        break
                # if r == 0:
                #     print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                #     continue        
                print_log('threshold for selecting:{}'.format(np.linalg.norm(activation)**2), log)
                print_log("total ranking r = {}".format(r), log)

                #=5 save the corresponding space
                Ui = np.hstack((space_list_all[i],U))
                sel_index = stack_index[:r]
                #print('sel_index:{}'.format(sel_index))
                # this is the current space
                U_new = Ui[:, sel_index]
                # calculate how many space from current new task
                sel_index_from_U = sel_index[sel_index>r_pre]
                # print(sel_index)
                # print(sel_index_from_U)
                if len(sel_index_from_U) > 0:
                    # update the overall space without overlap
                    total_U =  np.hstack((space_list_all[i], Ui[:,sel_index_from_U] ))

                    space_list_all[i] = total_U
                else:
                    space_list_all[i] = np.array(space_list_all[i])
 
                print_log("the number of space for current task:{}".format(r), log)
                print_log('the new increased space:{}, the threshold for new space:{}'.format(len(sel_index_from_U), r_pre), log)

                memory[task_name][str(i)]['space_list'] = Ui[:,sel_index]

            else:
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-8)
                # Go through all the previous tasks
                act_hat = activation
                for task_index in range(task_id):
                    space_list = memory[task_name_list[task_index]][str(i)]['space_list']
                    act_hat -= np.dot(np.dot(space_list,space_list.transpose()),activation)
                
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)

                    
                #update GPM
                # criteria (Eq-9)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total
                
                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < threshold[i]:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print_log('Skip Updating GPM for layer: {}'.format(i+1), log) 

                feature_list = []
                for task_index in range(task_id):
                    space_list = memory[task_name_list[task_index]][str(i)]['space_list']
                    feature_list.append(space_list)
                
                Ui=np.hstack((space_list_all[i],U[:,0:r]))  
                print_log('Ui shape:{}'.format(Ui.shape), log)
                if Ui.shape[1] > Ui.shape[0] :
                    space_list_all[i]=Ui[:,0:Ui.shape[0]]
                else:
                    space_list_all[i]=Ui
                                                                                                    
                if r == 0:
                    memory[task_name][str(i)]['space_list'] = space_list
                    # print(memory[task_name][str(i)]['space_list'])
                else:
                    memory[task_name][str(i)]['space_list'] = U[:,0:r]
 

    print_log('-'*40, log)
    print_log('Gradient Constraints Summary', log)
    print_log('-'*40, log)

    for i in range(3):
        print ('Layer {} : {}/{}'.format(i+1,space_list_all[i].shape[1], space_list_all[i].shape[0]))
    print_log('-'*40, log)
    
    return space_list_all       



def grad_proj_cond(args, net, x, y, memory, task_name, task_id, task_name_list, device, optimizer, criterion):

    # calcuate the gradient for current task before training
    steps = 1
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:300] # Take random training samples
    example_data = x[b].view(-1,28*28)
    example_data, target = example_data.to(device), y[b].to(device)
    
    batch_list=[300,300,300] 
    grad_list=[] # list contains gradient of each layer
    act_key=list(net.act.keys())

    for i in range(steps):
        optimizer.zero_grad()  
        example_out  = net(example_data)

        loss = criterion(example_out, target)         
        loss.backward()  

        for k, (m,params) in enumerate(net.named_parameters()):
            if 'weight' in m:
                grad = params.grad.data.detach().cpu().numpy()
                grad_list.append(grad)


    # project on each task subspace
    gradient_norm_lists_tasks = []
    # ratio_tasks = []
    for task_index in range(task_id):
        projection_norm_lists = []
        
        # ratio_layers = []
        for i in range(len(grad_list)):  #layer
            space_list = memory[task_name_list[task_index]][str(i)]['space_list']
            print_log("Task:{}, layer:{}, space shape:{}".format(task_index, i,space_list.shape), log)
            # grad_list is the grad for current task
            projection = np.dot(grad_list[i], np.dot(space_list,space_list.transpose()))
 
            projection_norm = np.linalg.norm(projection)

            projection_norm_lists.append(projection_norm)
            gradient_norm = np.linalg.norm(grad_list[i]) 
            print_log('Task:{}, Layer:{}, project_norm:{}, threshold for regime 1:{}'.format(task_index, i, projection_norm, eplison_1 * gradient_norm), log)


            if projection_norm <= eplison_1 * gradient_norm:
                memory[task_name][str(i)]['regime'][task_index] = '1'
            else:

                memory[task_name][str(i)]['regime'][task_index] = '2'
        # ratio_tasks.append(ratio_layers)

        gradient_norm_lists_tasks.append(projection_norm_lists)
        for i in range(len(grad_list)):
            print_log('Layer:{}, Regime:{}'.format(i, memory[task_name][str(i)]['regime'][task_index]), log)  
    # select top-k related tasks according to the projection norm, k = 2 in general (k= 1 for task 2)
    print_log('-'*20, log)
    print_log('selected top-2 tasks:', log)
    if task_id == 1:
        for i in range(len(grad_list)): 
            memory[task_name][str(i)]['selected_task'] = [0]
    else:
        if task_id == 2:
            for layer in range(len(grad_list)):
                memory[task_name][str(layer)]['selected_task'] = [1]
                print_log('Layer:{}, selected task ID:{}'.format(layer, memory[task_name][str(layer)]['selected_task']), log)
        else:
            k = 2
  
            for layer in range(len(grad_list)):
                task_norm = []
                for t in range(len(gradient_norm_lists_tasks)):
                    norm = gradient_norm_lists_tasks[t][layer]
                    task_norm.append(norm)
                task_norm = np.array(task_norm)
                idx = np.argpartition(task_norm, -k)[-k:]
                memory[task_name][str(layer)]['selected_task'] = idx
                print_log('Layer:{}, selected task ID:{}'.format(layer, memory[task_name][str(layer)]['selected_task']), log)
    
    # return ratio_tasks 


def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ## Load PMNIST DATASET
    from dataloader import pmnist as pmd
    data,taskcla,inputsize=pmd.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix=np.zeros((10,10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    task_name_list = []
    memory = {}

    acc_list_all = []

    # ratios = []
    epochs_back = []
    for k,ncla in taskcla:
        
        # specify threshold hyperparameter
        threshold = np.array([0.95,0.99,0.99]) 
        task_name = data[k]['name']
        task_name_list.append(task_name)
        print_log('*'*100, log)
        print_log('Task {:2d} ({:s})'.format(k,data[k]['name']), log)
        print_log('*'*100, log)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        lr = args.lr 
        print_log ('-'*40, log)
        print_log ('Task ID :{} | Learning Rate : {}'.format(task_id, lr), log)
        print_log ('-'*40, log)
        
        if task_id==0:
            model = MLPNet(args.n_hidden, args.n_outputs).to(device)
            memory[task_name] = {}
            #memory[task_name]['regime'] = 10 * [0]
            print_log ('Model parameters ---', log)
            k = 0
            for k_t, (m, param) in enumerate(model.named_parameters()):
                if 'weight' in m:
                    print(k,m,param.shape)
                    # create the saved memory
                    memory[task_name][str(k)] = {
                        'space_list': {},
                        'grad_list': {},
                        'regime':{},
                    }
                    k += 1
            print_log ('-'*40, log)

            space_list_all =[]
            # coord_list = []
            normal_param = [
                param for name, param in model.named_parameters()
                if not 'scale' in name
            ] 

            optimizer = torch.optim.SGD([
                                        {'params': normal_param}
                                        ],
                                        lr=lr
                                        )
            acc_list = []
            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain, criterion)
                print_log('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)),log)
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion)
                acc_list.append(valid_acc)
                print_log(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),log)
                print()
            print(acc_list)
            acc_list_all.append(acc_list)
            # Test
            print_log ('-'*40, log)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion)
            print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)
            # Memory Update  
            mat_list, grad_list = get_representation_and_gradient (model, device, optimizer, criterion, task_id,  xtrain, ytrain)
            space_list_all = get_space_and_grad (model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all)
            
        else:
            memory[task_name] = {}

            k = 0
            for k_t, (m, params) in enumerate(model.named_parameters()):
                # create the saved memory
                if 'weight' in m:
                    
                    memory[task_name][str(k)] = {
                        'space_list': {},
                        'grad_list': {},
                        'space_mat_list':{},
                        'scale1':{},
                        'scale2':{},
                        'regime':{},
                        'selected_task':{},
                        # 'ratio':{},
                    }
                    k += 1
                #reinitialize the scale
                if 'scale' in m:
                    mask = torch.eye(params.size(0), params.size(1)).to(device)
                    params.data = mask
            normal_param = [
                param for name, param in model.named_parameters()
                if not 'scale' in name 
            ] 

            scale_param = [
                param for name, param in model.named_parameters()
                if 'scale' in name 
            ]
            optimizer = torch.optim.SGD([
                                        {'params': normal_param},
                                        {'params': scale_param, 'weight_decay': 0, 'lr':lr}
                                        ],
                                        lr=lr
                                        )
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(space_list_all)):
                 Uf=torch.Tensor(np.dot(space_list_all[i],space_list_all[i].transpose())).to(device)
                 print_log('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape), log)
                 feature_mat.append(Uf)


            #==1 gradient projection condition
            print_log('excute gradient projection condition', log)
            grad_proj_cond(args, model, xtrain, ytrain, memory, task_name, task_id, task_name_list, device, optimizer, criterion)

            # optimizer = optim.SGD(model.parameters(), lr=args.lr)
            print_log('-'*40, log)

            # select the regime 2, which need to learn scale
            space1 = [None, None, None]
            space2 = [None, None, None]
            
      
            for i in range(3): #layer
                for k, task_sel in enumerate(memory[task_name][str(i)]['selected_task']):  #task loop
                    if memory[task_name][str(i)]['regime'][task_sel] == '2' or memory[task_name][str(i)]['regime'][task_sel] == '3':
                        if k == 0:
                            # change the np array to torch tensor
                            space1[i]=torch.tensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                        else:
                            space2[i]=torch.tensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)

            if space1[0] is not None:
                print_log('space1 is not None!', log)
            if space2[1] is not None:
                print_log('space2 is not None!', log) 

            print_log ('-'*40, log)
            acc_list = []

            for epoch in range(1, args.n_epochs+1):

                clock0=time.time()
                train_projected_regime(args, model,device,xtrain, ytrain,optimizer,criterion,memory, task_name, task_name_list, task_id, feature_mat, space1=space1, space2=space2)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,  criterion, space1=space1, space2=space2)
                print_log('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)),log)
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, space1=space1, space2=space2)
                acc_list.append(valid_acc)
      
                print_log(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),log)
                print()

            print(acc_list)
            acc_list_all.append(acc_list)
            print(epochs_back)

            # Test 
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, space1=space1, space2=space2)
            print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)  

            # Memory Update  
            mat_list, grad_list = get_representation_and_gradient (model, device, optimizer, criterion, task_id, xtrain, ytrain)
            space_list_all = get_space_and_grad (model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all)
            # save the scale value to memory
            idx1 = 0
            idx2 = 0
            for m,params in model.named_parameters():
                if 'scale1' in m:
                    memory[task_name][str(idx1)]['scale1'] = params.data
                    idx1 += 1
                if 'scale2' in m:
                    memory[task_name][str(idx2)]['scale2'] = params.data
                    idx2 += 1
        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            # select the regime 2, which need to learn scale
            space1 = [None, None, None]
            space2 = [None, None, None]
            task_test = data[ii]['name']
            print_log('current testing task:{}'.format(task_test), log)
            if ii > 0:
                          
                for i in range(3):
                    for k, task_sel in enumerate(memory[task_test][str(i)]['selected_task']):
                        if memory[task_test][str(i)]['regime'][task_sel] == '2' or memory[task_name][str(i)]['regime'][task_sel] == '3':
                            if k == 0:
                                # change the np array to torch tensor
                                space1[i] = torch.tensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                                idx = 0
                                for m,params in model.named_parameters():
                                    if 'scale1' in m:
                                        params.data = memory[task_test][str(idx)]['scale1'].to(device)
                                        idx += 1
                            else:
                                space2[i] = torch.tensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                                idx = 0
                                for m,params in model.named_parameters():                               
                                    if 'scale2' in m:
                                        params.data = memory[task_test][str(idx)]['scale2'].to(device)
                                        idx += 1                           

            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion, space1=space1, space2=space2) 
            jj +=1
        print_log('Accuracies =', log)
        for i_a in range(task_id+1):
            print_log('\t', log)
            for j_a in range(acc_matrix.shape[1]):
                print_log('{:5.1f}% '.format(acc_matrix[i_a,j_a]), log, end='')
        # for i_a in range(task_id+1):
        #     print('\t',end='')
        #     for j_a in range(acc_matrix.shape[1]):
        #         print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
        #     print()
        # update task id 
        task_id +=1
        save_model(model, memory, args.savename)

    print_log('-'*50, log)
    # Simulation Results 
    print_log ('Task Order : {}'.format(np.array(task_list)), log)
    print_log ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()), log) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print_log ('Backward transfer: {:5.2f}%'.format(bwt), log)
    print_log('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000), log)
    print_log('-'*50, log)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]],
                      columns = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]])
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})

    plt.show()


def print_log(print_string, log, end=None):
    if end is not None:
        print("{}".format(print_string), end='')
    else:
        print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--pc_valid',default=0.1,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # Architecture
    parser.add_argument('--n_hidden', type=int, default=100, metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='NT',
                        help='number of tasks (default: 10)')
    parser.add_argument('--savename', type=str, default='save/P_MNIST/Ours/two_task_overlap',
                        help='save path')
    parser.add_argument('--log_path', type=str, default='save/P_MNIST/Ours/two_task_overlap/train.log',
                        help='save path')

    args = parser.parse_args()
    if not os.path.exists(args.savename):
       os.makedirs(args.savename)
    log = open(os.path.join(args.savename,
                            'log_seed_{}.txt'.format(args.seed)), 'w')
    print_log('='*100, log)
    print_log('Arguments =', log)
    for arg in vars(args):
        print_log('\t'+arg+': {}'.format(getattr(args,arg)), log)
    print_log('='*100, log)


    print_log('save path : {}'.format(args.savename), log)
    main(args)




