from error_and_resample import *
import scipy
import gvar as gv
import time
import numpy as np


def print_fit(fit_):
    print('---------------------------------------------------')
    print('chi2/dof =', '%4.2f' % (fit_.chi2/fit_.dof), end='\t')
    print('\t dof =', '%d' % fit_.dof, end='\t')
    print('\t Q =', '%4.2f' % (scipy.special.gammaincc((fit_.dof-fit_.p.size)/2, fit_.chi2/2)))
    print('---------------------------------------------------')
    for key in fit_.p:
        print(key+' =', fit_.p[key])
    print('---------------------------------------------------')


def bs_fitting(fit_, nbs_, seed_=0):
    print('\nBootstrap Analysis with n_boot %d ...' % nbs_)
    st = time.time()
    gv.ranseed(seed_)
    n_boot = nbs_
    bs_res = {}
    for key in fit_.p:
        bs_res[key] = []
    for bs_fit in fit_.bootstrap_iter(n_boot):
        p = bs_fit.pmean
        for key in fit_.p:
            bs_res[key].append(p[key])
    for key in fit_.p:
        bs_res[key] = np.array(bs_res[key])
        # print(key+' =', gv.gvar(np.average(bs_res[key]), np.std(bs_res[key])))
        print(key+' =', gv.gvar(get_p68_mean_and_error(bs_res[key], 0)))
    ed = time.time()
    print('Bootstrap Analysis done, %4.2f s used.' % (ed - st))
    return bs_res
