# This code is from


import numpy as np

from .min_norm_solvers_numpy import MinNormSolver


def pmol_search(multi_obj_fg, pref_vec, x=None,
                max_iters=200, n_dim=20, step_size=1):
    """
    MOO-MTL
    """
    # x = np.random.uniform(-0.5,0.5,n_dim)
    x = np.random.randn(n_dim) if x is None else x
    Fs = []
    pref_vec = pref_vec / np.linalg.norm(pref_vec)
    
    init_lam_f = np.ones(2)/2
    init_lam_f = init_lam_f.reshape((2,1))
    init_lam_h = 0
    
    A = np.eye(2)
    # A = np.array([[3./np.sqrt(10), 1./np.sqrt(10)], 
    #               [0, 1]])
    # A = torch.from_numpy(A).float()
    
    
    Bh = np.array([pref_vec[1], -pref_vec[0]]).reshape((1, 2))
        
    
    for t in range(max_iters):
        F, grads = multi_obj_fg(x)
        weights, nd = get_d_pmol(grads, F, pref_vec)
        
        
        # lam = np.dot(A.T, init_lam_f) + np.dot(Bh.T, init_lam_h)
        # grad_lamt = np.dot(grads.T, lam)
        
        # weights, nd, lam_f, lam_h = get_d_pmol_single(grads, F, grad_lamt, 
        #             init_lam_f, init_lam_h, 
        #             pref_vec, 1)
        
        # init_lam_f = lam_f
        # init_lam_h = lam_h
        
        x = x - step_size * nd.flatten()                

        Fs.append(F)

    res = {'ls': np.stack(Fs)}
    return x, res


def get_d_pmol(grads, F, pref_vec):
    """
    calculate the gradient direction for FERERO
    """

    nobj, dim = grads.shape
    if nobj <= 1:
        return np.array([1.])
    
    A = np.eye(nobj)

    if nobj == 2:
        Bh = np.array([pref_vec[1], -pref_vec[0]]).reshape((1, nobj))
    
    H = Bh @ F 

    sol, nd = MinNormSolver.find_min_norm_element_PGD_H(grads, A, Bh, H)
   
    return sol, nd


def get_d_pmol_single(grads, F, grad_lamt, 
            init_lam_f, init_lam_h, 
            pref_vec, iter_K):
    """
    calculate the gradient direction for FERERO
    """
    nobj, dim = grads.shape
    if nobj <= 1:
        return np.array([1.])
    
    A = np.eye(nobj)
    
    # A = torch.from_numpy(A).float()
    
    if nobj == 2:
        Bh = np.array([pref_vec[1], -pref_vec[0]]).reshape((1, nobj))
    # Bh = torch.from_numpy(Bh).float()
    H = Bh @ F
    sol, nd, lam_f, lam_h = MinNormSolver.find_min_norm_element_PGD_H_single(
            grads, grad_lamt, init_lam_f, init_lam_h, 
            A, Bh, H, iter_K)

    return sol, nd, lam_f, lam_h
