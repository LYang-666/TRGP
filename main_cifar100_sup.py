import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
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
from layers import Conv2d, Linear
## Define LeNet model 
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class LeNet(nn.Module):
    def __init__(self,taskcla):
        super(LeNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        
        self.map.append(32)
        self.conv1 = Conv2d(3, 20, 5, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(3)        
        self.map.append(s)
        self.conv2 = Conv2d(20, 50, 5, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(20)        
        self.smid=s
        self.map.append(50*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(3,2,padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)
        self.lrn = torch.nn.LocalResponseNorm(4,0.001/9.0,0.75,1)

        self.fc1 = nn.Linear(50*self.smid*self.smid,800, bias=False)
        self.fc2 = nn.Linear(800,500, bias=False)
        self.map.extend([800])
        
        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(500,n,bias=False))
        
    def forward(self, x, space1= [None, None, None, None], space2= [None, None, None, None]):
        bsz = deepcopy(x.size(0))
        if space1[0] is not None or space2[0] is not None:
            self.act['conv1']=x
            x = self.conv1(x, space1=space1[0], space2 = space2[0])
            x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

            self.act['conv2']=x
            x = self.conv2(x, space1=space1[1], space2 = space2[1])
            x = self.maxpool(self.drop1(self.lrn (self.relu(x))))

            x=x.reshape(bsz,-1)
            self.act['fc1']=x
            x = self.fc1(x)
            x = self.drop2(self.relu(x))

            self.act['fc2']=x        
            x = self.fc2(x)
            x = self.drop2(self.relu(x))

            y=[]
            for t,i in self.taskcla:
                y.append(self.fc3[t](x))        
        else:
            self.act['conv1']=x
            x = self.conv1(x)
            x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

            self.act['conv2']=x
            x = self.conv2(x)
            x = self.maxpool(self.drop1(self.lrn (self.relu(x))))

            x=x.reshape(bsz,-1)
            self.act['fc1']=x
            x = self.fc1(x)
            x = self.drop2(self.relu(x))

            self.act['fc2']=x        
            x = self.fc2(x)
            x = self.drop2(self.relu(x))

            y=[]
            for t,i in self.taskcla:
                y.append(self.fc3[t](x))
            
        return y

def init_weights(m):
    # print(m)
    if isinstance(m, Linear) or isinstance(m, Conv2d) or isinstance(m, nn.Linear):
    # if type(m) == Linear or type(m) == Conv2d:
        print(m)
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
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
def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor  

def train(args, model, device, x,y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)        
        loss.backward()
        optimizer.step()

def train_projected_regime (args, model,device,x,y,optimizer,criterion,memory, task_name, task_name_list, space_list_all, task_id, feature_mat, space1=[None, None, None], space2=[None, None, None]):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data, space1=space1, space2=space2)
        loss = criterion(output[task_id], target)         
        loss.backward()        
        
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):

            if 'weight' in m:
                if k<8 and len(params.size())!=1:
                    sz =  params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                            feature_mat[kk]).view(params.size())
                    kk +=1
                elif (k<8 and len(params.size())==1) and task_id !=0 :
                    params.grad.data.fill_(0)
        

        optimizer.step()

            

def test(args, model, device, x, y, criterion, task_id, space1=[None, None, None], space2=[None, None, None]):
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
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data,space1=space1, space2=space2)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def get_representation_and_gradient (net, device, optimizer, criterion, task_id, x, y=None): 
    # Collect activations by forward pass
    steps = 1
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125 random samples 
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)
    example_out  = net(example_data)
    
    batch_list=[2*12,100,125,125] 
    pad = 2
    p1d = (2, 2, 2, 2)
    mat_list=[]
    act_key=list(net.act.keys())
    grad_list=[] # list contains gradient of each layer

    net.eval()
    example_out  = net(example_data)

    # pdb.set_trace()
    for i in range(len(net.map)):
        bsz=batch_list[i]
        k=0
        if i<2:
            ksz= net.ksize[i]
            s=compute_conv_output_size(net.map[i],net.ksize[i],1,pad)
            mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],s*s*bsz))
            act = F.pad(net.act[act_key[i]], p1d, "constant", 0).detach().cpu().numpy()
         
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) #?
                        k +=1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)


    print_log('-'*30, log)
    print_log('Representation Matrix', log)
    print_log('-'*30, log)
    for i in range(len(mat_list)):
        print_log ('Layer {} : {}'.format(i+1,mat_list[i].shape), log)
    print_log('-'*30, log)
    return mat_list, grad_list


def get_space_and_grad(model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all):
    print_log ('Threshold:{}'.format(threshold), log) 
    Ours = True
    if task_name == 'cifar100-0':
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            # gradient = grad_list[i]

            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  

            # save into memory
            memory[task_name][str(i)]['space_list'] = U[:,0:r]
            # memory[task_name][str(i)]['grad_list'] = gradient

            space_list_all.append(U[:,0:r])

    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]

            if Ours:
                #=1. calculate the projection using previous space
                print_log('activation shape:{}'.format(activation.shape), log)
                print_log('space shape:{}'.format(space_list_all[i].shape), log)
                #delta = np.dot(np.dot(space_list_all[i],space_list_all[i].transpose()),activation)
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
                # this is the current space
                U_new = Ui[:, sel_index]
                # calculate how many space from current new task
                sel_index_from_U = sel_index[sel_index>r_pre]
  
                if len(sel_index_from_U) > 0:
                    # update the overall space without overlap
                    total_U =  np.hstack((space_list_all[i], Ui[:,sel_index_from_U] ))
                    space_list_all[i] = total_U
                else:
                    space_list_all[i] = np.array(space_list_all[i])                

                print_log("the number of space for current task:{}".format(r), log)
                print_log('the new increased space:{}, the threshold for new space:{}'.format(len(sel_index_from_U), r_pre), log)
     
                print_log("Ui shape:{}".format(Ui[:,sel_index].shape), log)
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
                    print_log ('Skip Updating GPM for layer: {}'.format(i+1), log) 


                
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

    for i in range(4):
        print_log ('Layer {} : {}/{}'.format(i+1,space_list_all[i].shape[1], space_list_all[i].shape[0]), log)
    print_log('-'*40, log)
    
    return space_list_all      

def grad_proj_cond(net, x, y, memory, task_name, task_id, task_name_list, device, optimizer, criterion):
    eplison_1 = 0.5
    eplison_2 = 0.7

    # calcuate the gradient for current task before training
    steps = 1
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125*10 random samples
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)
    
    batch_list=[2*12,100,125,125] 
    grad_list=[] # list contains gradient of each layer
    act_key=list(net.act.keys())
    for i in range(1):


        optimizer.zero_grad()  
        example_out  = net(example_data)

        loss = criterion(example_out[task_id], target)         
        loss.backward()  

        k_linear = 0
        for k, (m,params) in enumerate(net.named_parameters()):
            if 'weight' in m and 'bn' not in m:
                if len(params.shape) == 4:
                    
                    grad = params.grad.data.detach().cpu().numpy()
                    grad = grad.reshape(grad.shape[0], grad.shape[1]*grad.shape[2]*grad.shape[3])
                    grad_list.append(grad)
                else:
                    if 'fc3' in m and k_linear == task_id:
                        grad = params.grad.data.detach().cpu().numpy()
                        grad_list.append(grad)
                        k_linear += 1
                    elif 'fc3' not in m:
                        grad = params.grad.data.detach().cpu().numpy()
                        grad_list.append(grad)   

    # project on each task subspace
    #gradient_norm_lists = []
    gradient_norm_lists_tasks = []

    for task_index in range(task_id):
        projection_norm_sum = 0
        projection_norm_lists = []
        # projection_norm = 0
        # gradient_norm = 0
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
      
        gradient_norm_lists_tasks.append(projection_norm_lists)
        for i in range(len(grad_list)):
            print_log('Layer:{}, Regime:{}'.format(i, memory[task_name][str(i)]['regime'][task_index]), log)  
    # select top-k related tasks according to the projection norm, k = 2 in general (k= 1 for task 2)
    if task_id == 1:
        for i in range(len(grad_list)): 
            memory[task_name][str(i)]['selected_task'] = [0]
    else:
        k = 2
        for layer in range(len(grad_list)):
            task_norm = []
            for t in range(len(gradient_norm_lists_tasks)):
                norm = gradient_norm_lists_tasks[t][layer]
                task_norm.append(norm)
            task_norm = np.array(task_norm)
            print(task_norm.shape)
            idx = np.argpartition(task_norm, -k)[-k:]
            memory[task_name][str(layer)]['selected_task'] = idx
            print_log('Layer:{}, selected task ID:{}'.format(layer, memory[task_name][str(layer)]['selected_task']), log)

def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Choose any task order - ref {yoon et al. ICLR 2020}
    task_order = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                  np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
                  np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
                  np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
                  np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])]

    ## Load CIFAR100_SUPERCLASS DATASET
    from dataloader import cifar100_superclass as data_loader
    data, taskcla = data_loader.cifar100_superclass_python(task_order[args.t_order], group=5, validation=True)
    test_data,_   = data_loader.cifar100_superclass_python(task_order[args.t_order], group=5)
    print (taskcla)

    acc_matrix=np.zeros((20,20))
    criterion = torch.nn.CrossEntropyLoss()
    task_id = 0
    task_list = []
    task_name_list = []
    memory = {}
    for k,ncla in taskcla:
        # specify threshold hyperparameter

        threshold = np.array([0.98] * 5) + task_id*np.array([0.001] * 5)

        task_name = data[k]['name']+'-'+str(k)
        task_name_list.append(task_name)
        print_log('*'*100, log)
        print_log('Task {:2d} ({:s})'.format(k,task_name), log)
        print_log('*'*100, log)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =test_data[k]['test']['x']
        ytest =test_data[k]['test']['y']
        task_list.append(k)

        lr = args.lr 
        best_loss=np.inf
        print_log ('-'*40, log)
        print_log ('Task ID :{} | Learning Rate : {}'.format(task_id, lr), log)
        print_log ('-'*40, log)

        if task_id==0:
            model = LeNet(taskcla).to(device)
            for k_t, (m, param) in enumerate(model.named_parameters()):
                print (k_t,m,param.shape)
            memory[task_name] = {}

            print_log ('Model parameters ---', log)
            kk = 0
            for k_t, (m, param) in enumerate(model.named_parameters()):
                if 'weight' in m and 'bn' not in m:
                    print_log ((k_t,m,param.shape), log)
                    memory[task_name][str(kk)] = {
                        'space_list': {},
                        'grad_list': {},
                        'regime':{},
                    }
                    kk += 1

            print_log ('-'*40, log)
            model.apply(init_weights)
            best_model=get_model(model)
            space_list_all =[]
            normal_param = [
                param for name, param in model.named_parameters()
                if not 'scale' in name
            ] 
            optimizer = torch.optim.SGD([
                                        {'params': normal_param}
                                        ],
                                        lr=lr
                                        )

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, k)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain, criterion, k)
                print_log('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)),log)
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, k)
                print_log(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),log)
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    print_log(' *',log)
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        print_log(' lr={:.1e}'.format(lr), log)
                        if lr<args.lr_min:
                            print_log("", log)
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print_log("", log)
            set_model_(model,best_model)
            # Test
            print_log ('-'*40, log)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)
            # Memory Update  
            mat_list, grad_list = get_representation_and_gradient (model, device, optimizer, criterion, k, xtrain, ytrain)
            space_list_all = get_space_and_grad (model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all)

        else:
            memory[task_name] = {}
            # memory[task_name]['regime'] = 10 * [0]
            #memory[task_name]['selected_task'] = []

            kk = 0
            print_log("reinit the scale for each task", log)
            for k_t, (m, params) in enumerate(model.named_parameters()):
                # create the saved memory
                if 'weight' in m and 'bn' not in m:
                    
                    memory[task_name][str(kk)] = {
                        'space_list': {},
                        'grad_list': {},
                        'space_mat_list':{},
                        'scale1':{},
                        'scale2':{},
                        'regime':{},
                        'selected_task':{}
                    }
                    kk += 1
                #reinitialize the scale
                if 'scale' in m:
                    mask = torch.eye(params.size(0), params.size(1)).to(device)

                    params.data = mask
            # print("-----------------")
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
            #Projection Matrix Precomputation
            for i in range(len(model.act)):
                Uf=torch.Tensor(np.dot(space_list_all[i],space_list_all[i].transpose())).to(device)
                print_log('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape), log)
                feature_mat.append(Uf)


            #==1 gradient projection condition
            print_log('excute gradient projection condition', log)
            grad_proj_cond(model, xtrain, ytrain, memory, task_name, task_id, task_name_list, device, optimizer, criterion)

            # select the regime 2, which need to learn scale
            space1 = [None, None, None, None]
            space2 = [None, None, None, None]
            for i in range(2):
                for k, task_sel in enumerate(memory[task_name][str(i)]['selected_task']):
                    if memory[task_name][str(i)]['regime'][task_sel] == '2' or memory[task_name][str(i)]['regime'][task_sel] == '3':
                        if k == 0:
                            # change the np array to torch tensor
                            space1[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                        else:
                            space2[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
            if space1[0] is not None:
                print_log('space1 is not None!', log)
            if space2[0] is not None:
                print_log('space2 is not None!', log) 

            print_log ('-'*40, log)
            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_projected_regime(args, model,device,xtrain, ytrain,optimizer,criterion,memory, task_name, task_name_list,space_list_all, task_id, feature_mat, space1=space1, space2=space2)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,criterion,task_id, space1=space1, space2=space2)
                print_log('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)),log)
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid, criterion, task_id, space1=space1, space2=space2)
                print_log(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),log)
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    print_log(' *',log)
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        print_log(' lr={:.1e}'.format(lr), log)
                        if lr<args.lr_min:
                            print_log("", log)
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print_log("", log)
            set_model_(model,best_model)
            # Test 
            test_acc_sum = 0
            # for i in range(10):
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion,task_id, space1=space1, space2=space2)
            print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)  
            # test_acc_sum = test_acc_sum/10.
            print_log('Average acc={:5.1f}%'.format(test_acc), log)  
            # Memory Update 
            mat_list, grad_list = get_representation_and_gradient (model, device, optimizer, criterion, task_id, xtrain, ytrain)
            space_list_all = get_space_and_grad (model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all)
            # save the scale value to memory
            idx1 = 0
            idx2 = 0
            for m,params in model.named_parameters(): # layer 
                if 'scale1' in m:
                    memory[task_name][str(idx1)]['scale1'] = params.data
                    idx1 += 1
                if 'scale2' in m:
                    memory[task_name][str(idx2)]['scale2'] = params.data
                    idx2 += 1          
        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =test_data[ii]['test']['x']
            ytest =test_data[ii]['test']['y'] 
            # select the regime 2, which need to learn scale
            space1 = [None, None, None, None]
            space2 = [None, None, None, None]
            task_test = task_name_list[ii]
            # print_log('current testing task:{}'.format(task_test), log)


            if ii > 0:
                for i in range(2):
                    for k, task_sel in enumerate(memory[task_test][str(i)]['selected_task']):
                        # print(memory[task_name]['regime'][task_sel])
                        if memory[task_test][str(i)]['regime'][task_sel] == '2' or memory[task_name][str(i)]['regime'][task_sel] == '3':
                            if k == 0:
                            # space1 = []
                                # change the np array to torch tensor
                                space1[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                                idx = 0
                                for m,params in model.named_parameters():
                                    if 'scale1' in m:
                                        params.data = memory[task_test][str(idx)]['scale1'].to(device)
                                        idx += 1
                            else:
                            #space2 = []
                                space2[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                                idx = 0
                                for m,params in model.named_parameters():                               
                                    if 'scale2' in m:
                                        params.data = memory[task_test][str(idx)]['scale2'].to(device)
                                        idx += 1   
         

            test_loss, test_acc = test(args, model, device, xtest, ytest,criterion,ii, space1=space1, space2=space2) 
            print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)  
            
            acc_matrix[task_id,jj] = test_acc
            #_, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion,ii, space1=space1, space2=space2) 
            jj +=1
        print_log('Accuracies =', log)
        for i_a in range(task_id+1):
            print_log('\t', log)
            for j_a in range(acc_matrix.shape[1]):
                print_log('{:5.1f}% '.format(acc_matrix[i_a,j_a]), log, end='')
            print_log("", log)
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
    df_cm = pd.DataFrame(array, index = [i for i in ["1","2","3","4","5","6","7",\
                                        "8","9","10","11","12","13","14","15","16","17","18","19","20"]],
                      columns = [i for i in ["1","2","3","4","5","6","7",\
                                        "8","9","10","11","12","13","14","15","16","17","18","19","20"]])
    sn.set(font_scale=1.4) 
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
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=50, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--t_order', type=int, default=0, metavar='TOD',
                        help='random seed (default: 0)')

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
    parser.add_argument('--savename', type=str, default='',
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


