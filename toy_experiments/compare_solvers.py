import numpy as np

import time

from problems.toy_biobjective import circle_points, concave_fun_eval, create_pf
from solvers import epo_search, pareto_mtl_search, linscalar, moo_mtl_search, pmol_search

import matplotlib.pyplot as plt
from latex_utils import latexify

if __name__ == '__main__':
    K = 4      # Number of trajectories
    n = 20      # dim of solution space
    m = 2       # dim of objective space
    rs = circle_points(K)  # preference

    pmtl_K = 5
    pmtl_refs = circle_points(pmtl_K, 0, np.pi / 2)
    
    methods = ['LinScalar', 'EPO', 'PMTL', 'MOOMTL', 'FERERO']
    
    latexify(fig_width=2., fig_height=1.55)
    ss, mi = 0.1, 100
    pf = create_pf()
    
    for method in methods:
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.12, bottom=.12, right=.97, top=.97)
        ax.plot(pf[:, 0], pf[:, 1], lw=2, c='k', label='Pareto Front')
        last_ls = []
        x00 = 0.1 * np.random.randn(n)
        for k, r in enumerate(rs):
            
            # if k != 2:
            #     continue
            
            r_inv = 1. / r
            ep_ray = 1.1 * r_inv / np.linalg.norm(r_inv)
            ep_ray_line = np.stack([np.zeros(m), ep_ray])

            label = r'$r^{-1}$ ray' if k == 0 else ''
            if method not in ["PCMG"]:
                
                ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color='k',
                    lw=1, ls='--', dashes=(15, 5), label=label)
                # ax.arrow(.95 * ep_ray[0], .95 * ep_ray[1],
                #      .05 * ep_ray[0], .05 * ep_ray[1],
                #      color='k', lw=1, head_width=.02)
            # x0 = np.random.randn(n) * 0.4
            
            x0 = np.zeros(n)
            x0[range(0, n, 2)] = 0.3
            x0[range(1, n, 2)] = -.3
            if method not in ["PCMG"]:
                x0 += 0.1 * np.random.randn(n)
            else:
                x0 += x00
            
            f0, _ = concave_fun_eval(x0)
            ax.scatter(f0[0], f0[1], s=40, c='g', alpha=1)
            
            if method in ["PCMG"]:
                
                ep_ray0 = f0 - 0.8 * ep_ray
                ep_ray_line0 = np.stack([f0, ep_ray0])
                ax.plot(ep_ray_line0[:, 0], ep_ray_line0[:, 1], color='k',
                    lw=1, ls='--', dashes=(15, 5), label=label)
                
                ax.arrow( ep_ray_line0[1, 0],  ep_ray_line0[1, 1],
                      -0.05 * ep_ray[0], -0.05 * ep_ray[1],
                      color='k', lw=1, head_width=.02)
            
            # if method not in ["PCMG"] else x0
            # x0 += 0.1 * np.random.randn(n) if method in ["PCMG"] else x0
            
            x0 = np.random.uniform(-0.6, 0.6, n) if method in [
                "MOOMTL", "LinScalar"] else x0
            
            
            time0 = time.time()
            if method == 'EPO':
                _, res = epo_search(concave_fun_eval, r=r, x=x0,
                                    step_size=ss, max_iters=80)
            if method == 'PMTL':
                _, res = pareto_mtl_search(concave_fun_eval,
                                           ref_vecs=pmtl_refs, r=r_inv, x=x0,
                                           step_size=0.2, max_iters=200)
            if method == 'LinScalar':
                _, res = linscalar(concave_fun_eval, r=r, x=x0,
                                    step_size=ss, max_iters=mi)
            if method == 'MOOMTL':
                _, res = moo_mtl_search(concave_fun_eval, x=x0,
                                        step_size=0.2, max_iters=150)
            
            if method == 'FERERO':
                _, res = pmol_search(concave_fun_eval, pref_vec=r_inv, x=x0,
                                    step_size=0.5, max_iters=10)
            rtime = time.time() - time0
            print(rtime)

            last_ls.append(res['ls'][-1])
            
            ls = res['ls']
            ax.plot(ls[:, 0], ls[:, 1], lw=1.5, c='b', alpha=0.7)
        
        last_ls = np.stack(last_ls)
        ax.scatter(last_ls[:, 0], last_ls[:, 1], s=40, c='b', alpha=1)
        ax.set_xlim(-0.1, 1.15)
        ax.set_ylim(-0.1, 1.15)
        ax.set_xticks(np.arange(0, 1.2, 0.2))
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_xlabel(r'$f_1$')
        ax.set_ylabel(r'$f_2$')
        
        # ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6, 2])
        ax.xaxis.set_label_coords(1.015, -0.03)
        ax.yaxis.set_label_coords(-0.01, 1.01)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        fig.savefig('figures/' + method + '.pdf')

    plt.show()
