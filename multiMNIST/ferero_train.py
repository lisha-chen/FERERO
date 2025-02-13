# FERERO
import numpy as np
import os
import math

from itertools import cycle
import torch
import torch.utils.data
from torch.autograd import Variable

from model_lenet import RegressionModel, RegressionTrain

from ferero_solvers import PMOLSolver
from time import time
import pickle

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # 
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def circle_points_(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles

def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def train(dataset, base_model, niter, pref_vec, init_weight):

    # generate #npref preference vectors      
    n_tasks = 2
    # ref_vec = torch.tensor(circle_points([1], [npref])[0]).cuda().float()
    
    A = np.eye(n_tasks)
    Bh = np.array([pref_vec[1], -pref_vec[0]]).reshape((1, n_tasks))
    
    # print(Bh)
    # load dataset 

    # MultiMNIST: multi_mnist.pickle
    if dataset == 'mnist':
        with open('data/multi_mnist.pickle','rb') as f:
            trainX, trainLabel,testX, testLabel = pickle.load(f)  
    
    # MultiFashionMNIST: multi_fashion.pickle
    if dataset == 'fashion':
        with open('data/multi_fashion.pickle','rb') as f:
            trainX, trainLabel,testX, testLabel = pickle.load(f)  
    
    
    # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    if dataset == 'fashion_and_mnist':
        with open('data/multi_fashion_and_mnist.pickle','rb') as f:
            trainX, trainLabel,testX, testLabel = pickle.load(f)   

    trainX = torch.from_numpy(trainX.reshape(120000,1,36,36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()
    testX = torch.from_numpy(testX.reshape(20000,1,36,36)).float()
    testLabel = torch.from_numpy(testLabel).long()
    
    
    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set  = torch.utils.data.TensorDataset(testX, testLabel)
    
    
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)
    
    print('==>>> total training batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader))) 
    
    
    # define the base model 
    model = RegressionTrain(RegressionModel(n_tasks), init_weight)
   
    
    if torch.cuda.is_available():
        model.cuda()


    # choose different optimizer for different base model
    if base_model == 'lenet':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90], gamma=0.5)
    
    if base_model == 'resnet18':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                            milestones=[10,20], gamma=0.1)
    
    
    # store infomation during optimization
    weights = []
    task_train_losses = []
    train_accs = []
        
    # print the current preference vector
    
    print(pref_vec)
    
    # run niter epochs of FERERO 
    init_lam_f = np.ones(n_tasks) / n_tasks
    init_lam_h = 0
    init_lam = np.dot(A.T, init_lam_f) + np.dot(Bh.T, init_lam_h)

    for t in range(niter):
        
        # scheduler.step()      
        model.train()
        for (it, batch) in enumerate(train_loader):   

            X = batch[0]
            ts = batch[1]
            
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()
            
            # double sampling 
            X1 = X[0:int(batch_size/2),:,:,:]
            ts1 = ts[0:int(batch_size/2),:]
            X2 = X[int(batch_size/2):batch_size,:,:,:]
            ts2 = ts[int(batch_size/2):batch_size,:]
            
            # obtain and store the gradient 
            grads = {}
            losses_vec = []    
            losses_vec_numpy = []        
            
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X1, ts1) 
                
                losses_vec.append(task_loss[i].data)
                losses_vec_numpy.append(losses_vec[i].cpu().numpy())
                
                task_loss[i].backward()
                
                # compute gradients             
                grads[i] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(
                            param.grad.data.clone().flatten(), requires_grad=False))
            
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            iter_K = 1
            
            optimizer.zero_grad()
            task_loss2 = model(X2, ts2)
            weighted_loss = torch.sum(
                task_loss2 * torch.from_numpy(init_lam).cuda())
            
            weighted_loss.backward()
        
            # compute gradient with weight lamt            
            grad_lamt = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_lamt.append(Variable(
                        param.grad.data.clone().flatten(), requires_grad=False))
            grad_lamt = torch.cat(grad_lamt)

            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            # print(losses_vec.shape)
            weight_vec, nd, lam_f, lam_h = PMOLSolver.get_d_pmol(
                grads.cpu().numpy(), np.array(losses_vec_numpy), 
                grad_lamt.cpu().numpy(), 
                init_lam_f, init_lam_h, pref_vec, iter_K)
            
            # normalize the weight lam
            # normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
            normalize_coeff = 1. / np.sum(abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff
            
            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = model(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            
            loss_total.backward()
            optimizer.step()
            init_lam_f = lam_f
            init_lam_h = lam_h
            init_lam = weight_vec

        # calculate and record performance
        if t == 0 or (t + 1) % 2 == 0:
            
            model.eval()
            with torch.no_grad():
  
                total_train_loss = []
                train_acc = []
        
                correct1_train = 0
                correct2_train = 0
                
                for (it, batch) in enumerate(test_loader):
                   
                    X = batch[0]
                    ts = batch[1]
                    if torch.cuda.is_available():
                        X = X.cuda()
                        ts = ts.cuda()
        
                    valid_train_loss = model(X, ts)
                    total_train_loss.append(valid_train_loss)
                    output1 = model.model(X).max(2, keepdim=True)[1][:,0]
                    output2 = model.model(X).max(2, keepdim=True)[1][:,1]
                    correct1_train += output1.eq(
                        ts[:,0].view_as(output1)).sum().item()
                    correct2_train += output2.eq(
                        ts[:,1].view_as(output2)).sum().item()
                                        
                train_acc = np.stack(
                    [1.0 * correct1_train / len(test_loader.dataset),
                     1.0 * correct2_train / len(test_loader.dataset)])
        
                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim = 0)
                            
            # record and print
            if torch.cuda.is_available():
                
                task_train_losses.append(average_train_loss.data.cpu().numpy())
                train_accs.append(train_acc)
                
                weights.append(weight_vec)
                
                print('{}/{}: weights={}, train_loss={}, train_acc={}'.format(
                        t + 1, niter,  weights[-1], task_train_losses[-1], train_accs[-1]))                 
                
                # avoid nan or other errors
                if math.isnan(any(task_train_losses[-1])) or any(
                    train_accs[-1] < 0.15):                 
                    return -1      
                  
    # torch.save(model.model.state_dict(), 
    #            './saved_model/FERERO/FERERO_%s_%s_niter_%d.pickle'%(dataset, base_model, niter))
    result = {"training_losses": task_train_losses,
            "training_accuracies": train_accs}
    return result
    

def run(dataset = 'mnist',base_model = 'lenet', niter = 100, npref = 5):
    """
    run FERERO
    """
    
    init_weight = np.array([0.5 , 0.5 ])
    start_time = time()
    preferences = circle_points(npref, 
                min_angle=0.0001*np.pi/2, max_angle=0.9999*np.pi/2)  
    results = dict()
    out_file_prefix = f"ferero_{dataset}_{base_model}_{niter}_{npref}_from_0-"
    for i, pref in enumerate(preferences):
        if i == 0 or i == 4: # only use the 3 preferences for the figure and HV
            continue

        s_t = time()
        pref_idx = i 
        pref = pref / np.linalg.norm(pref)
        r_inv = np.sqrt(1 - pref ** 2)
        
        res = -1
        while res == -1:
            res = train(dataset, base_model, niter, r_inv, init_weight)
        
        results[i] = {"r": pref, "res": res}
        print(f"**** Time taken for {dataset}_{i} = {time() - s_t}")
        results_file = os.path.join("results", out_file_prefix + f"{i}.pkl")
        
        #  
        pickle.dump(results, open(results_file, "wb"))
    print(f"**** Time taken for {dataset} = {time() - start_time}")


run(dataset = 'mnist', base_model = 'lenet', niter = 100, npref = 5)
run(dataset = 'fashion', base_model = 'lenet', niter = 100, npref = 5)
run(dataset = 'fashion_and_mnist', base_model = 'lenet', niter = 100, npref = 5)



