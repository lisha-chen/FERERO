# PMOL

import numpy as np
import torch

from min_norm_solvers import MinNormSolver

class PMOLSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum+= sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    def _next_point(cur_val, grad, n):
        proj_grad = grad - ( np.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
        
        skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
        t = 1
        if len(tm1[tm1>1e-7]) > 0:
            t = np.min(tm1[tm1>1e-7])
        if len(tm2[tm2>1e-7]) > 0:
            t = min(t, np.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = PMOLSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = PMOLSolver._min_norm_2d(vecs, dps)
        
        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]
    
        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]
                

        while iter_count < PMOLSolver.MAX_ITER:
            grad_dir = -1.0*np.dot(grad_mat, sol_vec)
            new_point = PMOLSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
                    v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
                    v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
            nc, nd = PMOLSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < PMOLSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
    
    def find_min_norm_element_PGD(grads):
        """
        We directly run the PGD until convergence
        """
        M = len(grads)
        # uniform as init
        init_lam = np.ones(M) / M
           
        iter_count = 0
        lam = init_lam 
        while iter_count < PMOLSolver.MAX_ITER:
            new_lam = lam - np.dot(grads, np.dot(grads.T, lam)) 
            new_lam = PMOLSolver._projection2simplex(new_lam)
            
            change = new_lam - lam
            if np.sum(np.abs(change)) < PMOLSolver.STOP_CRIT:
                break
            else:
                lam = new_lam
                iter_count += 1
                continue
        sol_lam = new_lam   
        nd = np.dot(grads.T, sol_lam)
        return sol_lam, nd
    
    def find_min_norm_element_PGD_H(grads, grad_lamt, 
                        init_lam_f, init_lam_h, 
                        A, Bh, H, iter_K):
        """
        We directly run the PGD with K iterations or till convergence
        with equality constraints H
        """
        M = len(grads)
        # uniform as init
        init_lam_f = np.ones(M) / M
           
        iter_count = 0

        lam_f = init_lam_f 
        Mh = Bh.shape[0]
        lam_h = init_lam_h
        gradsA = (grads.T @ A.T).T
        gradsBh = (grads.T @ Bh.T).T
        
        lam = np.dot(A.T, lam_f) + np.dot(Bh.T, lam_h)
        
        gamma = 1e-5
        while iter_count < iter_K:
            new_lam_f = lam_f - gamma * np.dot(gradsA, grad_lamt) 
            new_lam_f = PMOLSolver._projection2simplex(new_lam_f)
            
            new_lam_h = lam_h - gamma * (np.dot(
                gradsBh, grad_lamt) - 0.5 * H)            
            new_lam = np.dot(A.T, new_lam_f) + np.dot(Bh.T, new_lam_h)

            change = np.sum(np.abs(new_lam - lam))
            if change < PMOLSolver.STOP_CRIT:
                break
            else:
                lam_f = new_lam_f
                lam_h = new_lam_h
                lam = new_lam
                iter_count += 1
                continue
        
        nd_tplus = np.dot(grads.T, lam)
        return lam, nd_tplus, lam_f, lam_h
    
    def get_d_pmol(grads, F, grad_lamt, 
                init_lam_f, init_lam_h, 
                pref_vec, iter_K):
        """
        calculate the gradient direction for PMOL
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
        sol, nd, lam_f, lam_h = PMOLSolver.find_min_norm_element_PGD_H(
                grads, grad_lamt, init_lam_f, init_lam_h, 
                A, Bh, H, iter_K)

        return sol, nd, lam_f, lam_h

    
def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum(
                [gr.pow(2).sum().data[0] for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum(
                [gr.pow(2).sum().data[0] for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn