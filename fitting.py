from error_and_resample import *
import scipy
import gvar as gv
import time
import numpy as np
import lsqfit
import xarray as xr


def print_fit(fit_):
    print('-------------------------------------------------------------------------')
    print('chi2/dof =', '%4.2f' % (fit_.chi2/fit_.dof), end='\t')
    if (fit_.chi2/fit_.dof) > 2.:
        print("WARNING, large chi2")
    print('\t dof =', '%d' % fit_.dof, end='\t')
    print('\t Q =', '%4.2f' % (scipy.special.gammaincc((fit_.dof-fit_.p.size)/2, fit_.chi2/2)))
    print('--------------------------------------------------------------------------')
    for key in fit_.p:
        print(key+' =', fit_.p[key])
    print('--------------------------------------------------------------------------')


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
        print(p)
        for key in fit_.p:
            bs_res[key].append(p[key])
    for key in fit_.p:
        bs_res[key] = np.array(bs_res[key])
        print(key+' =', gv.gvar(get_p68_mean_and_error(bs_res[key])))
    ed = time.time()
    print('Bootstrap Analysis done, %4.2f s used.' % (ed - st))
    return bs_res


def jackknife_fitting_prior(data_, func_, prior0_, ncut1, ncut2, t1, t2):
    jk_res = []
    for ij in range(data_.shape[0]):
        data_new_ = np.delete(data_, ij, 0)
        data_fit = prep_data_ratio(data_new_,ncut1,ncut2,t1,t2)
        fit = lsqfit.nonlinear_fit(data=data_fit, prior=prior0_, fcn=func_, debug=True)
        jk_res.append(fit.p)
    return jk_res


def bootstrap_fitting_prior(nbs_, data_, func_, prior0_, ncut1, ncut2, t1, t2):
    bs_res = []
    for ib in range(nbs_):
        xb_ = get_boot_sample(np.arange(0, data_.shape[0], 1))
        data_new_ = data_[xb_, ...]
        data_fit = prep_data_ratio(data_new_,ncut1,ncut2,t1,t2)
        fit = lsqfit.nonlinear_fit(data=data_fit, prior=prior0_, fcn=func_, debug=True)
        bs_res.append(fit.p)
    return bs_res


def bootstrap_fitting(nbs_, data_, func_, p0_, ncut1, ncut2, t1, t2):
    bs_res = []
    for ib in range(nbs_):
        xb_ = get_boot_sample(np.arange(0, data_.shape[0], 1))
        data_new_ = data_[xb_, ...]
        data_fit = prep_data_ratio(data_new_,ncut1,ncut2,t1,t2)
        fit = lsqfit.nonlinear_fit(data=data_fit, p0=p0_, fcn=func_, debug=True)
        bs_res.append(fit.p)
    return bs_res


def prep_data_ratio(data, ncut1, ncut2, t1, t2):  #TODO
    nconf = data.shape[0]
    data_ave = np.average(data, 0)
    data_cov = np.cov(data, rowvar=False) / (nconf - 1) #FIXME
    data_gvr = gv.gvar(data_ave, data_cov).reshape(t2.size, t1.size)
    t = []
    y = []
    for it2 in range(t2.size):
        for it1 in range(ncut1, t2[it2] - ncut2):
            t.append((it1, t2[it2]))
            y.append(data_gvr[it2, it1])
    t = np.array(t)
    y = np.array(y)
    return (t, y)


def prep_data_ratio_new(data, ncut1, ncut2, t, t2):  #TODO
    nconf = data.shape[0]
    data_ave = np.average(data, 0)
    data_cov = np.cov(data, rowvar=False) / (nconf - 1) #FIXME
    data_gvr = gv.gvar(data_ave, data_cov).reshape(t.size, t2.size)
    t_ = []
    y_ = []
    for it in range(t.size):
        for it2 in range(ncut1, t[it] - ncut2 + 1):
            t_.append((it2, t[it]))
            y_.append(data_gvr[it, it2])
    t_ = np.array(t_)
    y_ = np.array(y_)
    return (t_, y_)


def prep_data_ratio_new_boot(data, ncut1, ncut2, t, t2):  #TODO
    nconf = data.shape[0]
    data_ave = np.average(data, 0)
    data_cov = np.cov(data, rowvar=False) #FIXME
    data_gvr = gv.gvar(data_ave, data_cov).reshape(t.size, t2.size)
    t_ = []
    y_ = []
    for it in range(t.size):
        for it2 in range(ncut1, t[it] - ncut2 + 1):
            t_.append((it2, t[it]))
            y_.append(data_gvr[it, it2])
    t_ = np.array(t_)
    y_ = np.array(y_)
    return (t_, y_)

def prep_data_ratio_new_jack(data, ncut1, ncut2, t, t2):  #TODO
    nconf = data.shape[0]
    data_ave = np.average(data, 0)
    data_cov = np.cov(data, rowvar=False) * (nconf - 1) #FIXME
    data_gvr = gv.gvar(data_ave, data_cov).reshape(t.size, t2.size)
    t_ = []
    y_ = []
    for it in range(t.size):
        for it2 in range(ncut1, t[it] - ncut2 + 1):
            t_.append((it2, t[it]))
            y_.append(data_gvr[it, it2])
    t_ = np.array(t_)
    y_ = np.array(y_)
    return (t_, y_)



def prep_data_sum(data, ncut1, ncut2, t, t2):  #TODO
    nconf = data.shape[0]
    data_ave = np.average(data, 0)
    data_cov = np.cov(data, rowvar=False) / (nconf - 1) #FIXME
    data_gvr = gv.gvar(data_ave, data_cov).reshape(t.size, t2.size)
    t_ = []
    y_ = []
    for it in range(t.size):
        t_.append(t[it])
        y_.append(0.0)
        for it2 in range(ncut1, t[it] - ncut2):
            y_[it] += data_gvr[it, it2]
    t_ = np.array(t_)
    y_ = np.array(y_)
    return (t_, y_)


def prep_data_ratio_jack(data, ncut1, ncut2, t1, t2):  #TODO
    nconf = data.shape[0]
    data_ave = np.average(data, 0)
    data_cov = np.cov(data, rowvar=False) * (nconf - 1) #FIXME
    data_gvr = gv.gvar(data_ave, data_cov).reshape(t2.size, t1.size)
    t = []
    y = []
    for it2 in range(t2.size):
        for it1 in range(ncut1, t2[it2] - ncut2):
            t.append((it1, t2[it2]))
            y.append(data_gvr[it2, it1])
    t = np.array(t)
    y = np.array(y)
    return (t, y)


def prep_data_ratio_jack2(data, ncut1, ncut2, t1, t2):  #TODO
    nconf = data.coords['jackknife'].size
    data_ave = data.transpose('jackknife','t2','t').lattice.mean('jackknife')
    data_err = data.transpose('jackknife','t2','t').lattice.jack_error()
    data_cov = np.cov(data.transpose('jackknife','t2','t').values.reshape(nconf,t2.size*t1.size), rowvar=False) * (nconf - 1) #FIXME
    data_gvr = gv.gvar(data_ave.values.reshape(t2.size*t1.size), data_cov).reshape(t2.size, t1.size)
    t = []
    y = []
    for it2 in range(t2.size):
        for it1 in range(ncut1, t2[it2] - ncut2 + 1):
            t.append((it1, t2[it2]))
            y.append(data_gvr[it2, it1])
    t = np.array(t)
    y = np.array(y)
    return (t, y)


def prep_data_ratio_boot(data, ncut1, ncut2, t1, t2):  #TODO
    nconf = data.shape[0]
    data_ave = np.average(data, 0)
    data_cov = np.cov(data, rowvar=False) #FIXME
    data_gvr = gv.gvar(data_ave, data_cov).reshape(t2.size, t1.size)
    t = []
    y = []
    for it2 in range(t2.size):
        for it1 in range(ncut1, t2[it2] - ncut2):
            t.append((it1, t2[it2]))
            y.append(data_gvr[it2, it1])
    t = np.array(t)
    y = np.array(y)
    return (t, y)

def prep_data_ratio_boot2(data, ncut1, ncut2, t1, t2):  #TODO
    nconf = data.coords['nothing'].size
    data_ave = data.transpose('nothing','t2','t').lattice.mean('nothing')
    data_err = np.std(data.transpose('nothing','t2','t'),0)
    data_cov = np.cov(data.transpose('nothing','t2','t').values.reshape(nconf,t2.size*t1.size), rowvar=False) #FIXME
    data_gvr = gv.gvar(data_ave.values.reshape(t2.size*t1.size), data_cov).reshape(t2.size, t1.size)
    t = []
    y = []
    for it2 in range(t2.size):
        for it1 in range(ncut1, t2[it2] - ncut2 + 1):
            t.append((it1, t2[it2]))
            y.append(data_gvr[it2, it1])
    t = np.array(t)
    y = np.array(y)
    return (t, y)


def fcn_2st_3t(t_, p_):
    c0_ = p_['c0']
    c1_ = p_['c1']
    c2_ = p_['c2']
    dm_ = p_['dm']
    return c0_ + c1_ * np.exp(-dm_ * (t_[:, 1] - t_[:, 0])) + c2_ * np.exp(-dm_ * t_[:, 0])


def fcn_2st_3t_share_c1(t_, p_):
    c0_ = p_['c0']
    c1_ = p_['c1']
    dm_ = p_['dm']
    return c0_ + c1_ * np.exp(-dm_ * (t_[:, 1] - t_[:, 0])) + c1_ * np.exp(-dm_ * t_[:, 0])


def fcn_2st_3t_plot(t1_, t2_, p_):
    c0_ = p_['c0'].mean 
    c1_ = p_['c1'].mean
    c2_ = p_['c2'].mean
    dm_ = p_['dm'].mean
    return c0_ + c1_ * np.exp(-dm_ * (t2_ - t1_)) + c2_ * np.exp(-dm_ * t1_)


def fcn_2st_4t(t_, p_):
    c0_ = p_['c0']
    c1_ = p_['c1']
    c2_ = p_['c2']
    c3_ = p_['c3']
    dm_ = p_['dm']
    return c0_ + c1_ * np.exp(-dm_ * (t_[:, 1] - t_[:, 0])) + c2_ * np.exp(-dm_ * t_[:, 0]) + c3_ * np.exp(-dm_ * t_[:, 1])


def fcn_2st_4t_plot(t1_, t2_, p_):
    c0_ = p_['c0'].mean 
    c1_ = p_['c1'].mean
    c2_ = p_['c2'].mean
    c3_ = p_['c3'].mean
    dm_ = p_['dm'].mean
    return c0_ + c1_ * np.exp(-dm_ * (t2_ - t1_)) + c2_ * np.exp(-dm_ * t1_) + c3_ * np.exp(-dm_ * t2_)


def fcn_const(t_, p_):
    c0_ = p_['c0']
    return c0_ + t_[:, 0] * 0.


def fcn_const2(t_, p_):
    c0_ = p_['c0']
    return c0_ + t_[:] * 0.

p0_2st_3t = {}
p0_2st_3t['c0'] = 2
p0_2st_3t['c1'] = 0.1
p0_2st_3t['c2'] = 0.1
p0_2st_3t['dm'] = 0.5


p0_2st_4t = {}
p0_2st_4t['c0'] = .1
p0_2st_4t['c1'] = -0.1
p0_2st_4t['c2'] = -0.1
p0_2st_4t['c3'] = 0.1
p0_2st_4t['dm'] = 0.5


prior0_2st_3t_share_c1 = {}
prior0_2st_3t_share_c1['c0'] = gv.gvar(-.05, np.inf)
prior0_2st_3t_share_c1['c1'] = gv.gvar(0.1, np.inf)
prior0_2st_3t_share_c1['dm'] = gv.gvar(0.3, 0.1)


prior0_2st_3t = {}
prior0_2st_3t['c0'] = gv.gvar(-.05, np.inf)
prior0_2st_3t['c1'] = gv.gvar(0.1, np.inf)
prior0_2st_3t['c2'] = gv.gvar(0.1, np.inf)
prior0_2st_3t['dm'] = gv.gvar(0.4, 0.01)


prior0_2st_4t = {}
prior0_2st_4t['c0'] = gv.gvar(.1, np.inf)
prior0_2st_4t['c1'] = gv.gvar(-0.1, np.inf)
prior0_2st_4t['c2'] = gv.gvar(-0.1, np.inf)
prior0_2st_4t['c3'] = gv.gvar(0.1, np.inf)
prior0_2st_4t['dm'] = gv.gvar(0.5, 0.5)


p0_const = {}
p0_const['c0'] = .1
    

def fcn_linear_m0(t_, p_):
    m0 = 0.139
    c0_ = p_['c0']
    c1_ = p_['c1']
    return c0_ + c1_ * (t_ - m0**2)


def fcn_linear(t_, p_):
    c0_ = p_['c0']
    c1_ = p_['c1']
    return c0_ + c1_ * t_


p0_linear = {}
p0_linear['c0'] = 2. 
p0_linear['c1'] = -.1
prior0_linear = {}
prior0_linear['c0'] = gv.gvar(.1, np.inf) 
prior0_linear['c1'] = gv.gvar(.1, np.inf)


def fcn_square(t_, p_):
    m0 = 0.139
    c0_ = p_['c0']
    c1_ = p_['c1']
    c2_ = p_['c2']
    return c0_ + c1_ * (t_ - m0**2) + c2_ * (t_**2 - m0**4)

def fcn_square_nom0(t_, p_):
    c0_ = p_['c0']
    c1_ = p_['c1']
    c2_ = p_['c2']
    return c0_ + c1_ * (t_) + c2_ * (t_**2)

p0_square = {}
p0_square['c0'] = .1
p0_square['c1'] = .1
p0_square['c2'] = .1

prior0_square = {}
prior0_square['c0'] = gv.gvar(.1, np.inf) 
prior0_square['c1'] = gv.gvar(.1, np.inf)
prior0_square['c2'] = gv.gvar(.1, np.inf)


def fcn_inverse(t_, p_):
    c0_ = p_['c0']
    c1_ = p_['c1']
    c2_ = p_['c2']
    return c1_/t_**2 + c0_  + c2_ * t_

p0_inverse = {}
p0_inverse['c0'] = .1
p0_inverse['c1'] = .1
p0_inverse['c2'] = .1

prior0_inverse = {}
prior0_inverse['c0'] = gv.gvar(.1, np.inf) 
prior0_inverse['c1'] = gv.gvar(.1, np.inf)
prior0_inverse['c2'] = gv.gvar(.1, np.inf)


def fcn_exp(t_, p_):
    c0_ = p_['c0']
    c1_ = p_['c1']
    return c0_ * np.exp(-c1_* t_)

p0_exp = {}
p0_exp['c0'] = .1
p0_exp['c1'] = .1

prior0_exp = {}
prior0_exp['c0'] = gv.gvar(.1, np.inf) 
prior0_exp['c1'] = gv.gvar(.1, np.inf)

def fcn_2exp(t_, p_):
    c0_ = p_['c0']
    c1_ = p_['c1']
    c2_ = p_['c2']
    c3_ = p_['c3']
    return c0_ * np.exp(-c1_* t_) + c2_ * np.exp(-(c1_+c3_) * t_)

p0_2exp = {}
p0_2exp['c0'] = .1
p0_2exp['c1'] = .1
p0_2exp['c2'] = 1
p0_2exp['c3'] = .0

prior0_2exp = {}
prior0_2exp['c0'] = gv.gvar(.1, np.inf) 
prior0_2exp['c1'] = gv.gvar(.1, np.inf)
prior0_2exp['c2'] = gv.gvar(.1, np.inf) 
prior0_2exp['c3'] = gv.gvar(.1, np.inf)



def do_2st_3t_fit(ratio_,t2_,ncut1_=1,ncut2_=0,dm_=gv.gvar(0.4, np.inf),show=True):
    nconf_ = ratio_.coords['conf'].size
    t_ = ratio_.coords['t'].values
    data_ = ratio_.sel(t2=t2_).squeeze().transpose('conf', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
    data_fit = prep_data_ratio_new(data_,ncut1_,ncut2_,t2_,t_)
    # print(data_fit[0].shape)
    # print(data_fit[1].shape)
    prior0_2st_3t['dm'] = dm_
    fit_ = lsqfit.nonlinear_fit(data=data_fit, prior=prior0_2st_3t, fcn=fcn_2st_3t, debug=True)
    if show:
        print_fit(fit_)
    return fit_


prior0_2st_3t_combine2={}
prior0_2st_3t_combine2['c0_1'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine2['c1_1'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine2['c2_1'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine2['c0_2'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine2['c1_2'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine2['c2_2'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine2['c3_1'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine2['c3_2'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine2['dm'] = gv.gvar(0.3,np.inf) 

def fcn_2st_3t_combine2(t_, p_):
    c01_ = p_['c0_1']
    c11_ = p_['c1_1']
    c21_ = p_['c2_1']
    c02_ = p_['c0_2']
    c12_ = p_['c1_2']
    c22_ = p_['c2_2']
    c31_ = p_['c3_1']
    c32_ = p_['c3_2']
    dm_ = p_['dm']
    ans = {}
    ans['1'] = c01_ + c11_ * np.exp(-dm_ * (t_['1'][:, 1] - t_['1'][:, 0])) + c21_ * np.exp(-dm_ * t_['1'][:, 0]) + c31_ * np.exp(-dm_ * t_['1'][:, 1])
    ans['2'] = c02_ + c12_ * np.exp(-dm_ * (t_['2'][:, 1] - t_['2'][:, 0])) + c22_ * np.exp(-dm_ * t_['2'][:, 0]) + c32_ * np.exp(-dm_ * t_['2'][:, 1])
    return ans

def do_2st_3t_fit_combine2(ratio1_,ratio2_, t2_,ncut1_=1,ncut2_=0,dm_=gv.gvar(0.4, np.inf),show=True,method=0):
    nconf_ = ratio1_.coords['conf'].size
    t_ = ratio1_.coords['t'].values

    if method == 1:
        data1_ = ratio1_.sel(t2=t2_).squeeze().transpose('conf', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
        data2_ = ratio2_.sel(t2=t2_).squeeze().transpose('conf', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
        data1_fit = prep_data_ratio_new(data1_,ncut1_,ncut2_,t2_,t_)
        data2_fit = prep_data_ratio_new(data2_,ncut1_,ncut2_,t2_,t_)
        x_temp={}
        x_temp['1'] = data1_fit[0]
        x_temp['2'] = data2_fit[0]
        data_temp = {}
        data_temp['1'] = data1_fit[1]
        data_temp['2'] = data2_fit[1]
    else:
        data_ = xr.concat([ratio1_, ratio2_], dim='operator')
        data_ = data_.squeeze().transpose('conf', 'operator','t2', 't').values.reshape(nconf_, 2 * t2_.size * t_.size) 
        data_ave = np.average(data_, 0)
        data_err = np.std(data_, 0) / np.sqrt(nconf_ - 1)
        data_cov = np.cov(data_, rowvar=False) / (nconf_ - 1) #FIXME
        data_gvr = gv.gvar(data_ave, data_cov).reshape(2, t2_.size, t_.size)
        if method == 0:
            data_gvr = gv.gvar(data_ave, data_err).reshape(2, t2_.size, t_.size)
        x_temp={}
        x_temp['1'] = [] 
        x_temp['2'] = [] 
        data_temp = {}
        data_temp['1'] = []
        data_temp['2'] = []
        for it2 in range(t2_.size):
            for it in range(ncut1_, t2_[it2] - ncut2_ + 1):
                x_temp['1'].append((it, t2_[it2]))
                x_temp['2'].append((it, t2_[it2]))
                data_temp['1'].append(data_gvr[0, it2, it])
                data_temp['2'].append(data_gvr[1, it2, it])
        x_temp['1'] = np.array(x_temp['1']) 
        x_temp['2'] = np.array(x_temp['2']) 
        data_temp['1'] = np.array(data_temp['1']) 
        data_temp['2'] = np.array(data_temp['2'])

    data_fit = (x_temp,data_temp)
    #print(data_fit)

    prior0_2st_3t_combine2['dm'] = dm_ 
    fit_ = lsqfit.nonlinear_fit(data=data_fit, prior=prior0_2st_3t_combine2, fcn=fcn_2st_3t_combine2, debug=True)
    if show:
        print_fit(fit_)
        print(fit_.p['c0_2']-fit_.p['c0_1'])
    return fit_

def do_2st_3t_fit_combine2_boot(ratio1_,ratio2_, t2_,ncut1_=1,ncut2_=0,dm_=gv.gvar(0.4, np.inf),show=True):
    nconf_ = ratio1_.coords['nothing'].size
    t_ = ratio1_.coords['t'].values
    data1_ = ratio1_.sel(t2=t2_).squeeze().transpose('nothing', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
    data2_ = ratio2_.sel(t2=t2_).squeeze().transpose('nothing', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
    data1_fit = prep_data_ratio_new_boot(data1_,ncut1_,ncut2_,t2_,t_)
    data2_fit = prep_data_ratio_new_boot(data2_,ncut1_,ncut2_,t2_,t_)
    x_temp={}
    x_temp['1'] = data1_fit[0]
    x_temp['2'] = data2_fit[0]
    data_temp = {}
    data_temp['1'] = data1_fit[1]
    data_temp['2'] = data2_fit[1]
    data_fit = (x_temp,data_temp)

    prior0_2st_3t_combine2['dm'] = dm_ 
    fit_ = lsqfit.nonlinear_fit(data=data_fit, prior=prior0_2st_3t_combine2, fcn=fcn_2st_3t_combine2, debug=True)
    if show:
        print_fit(fit_)
        print(fit_.p['c0_2']-fit_.p['c0_1'])
    return fit_

def do_2st_3t_fit_combine2_jack(ratio1_,ratio2_, t2_,ncut1_=1,ncut2_=0,dm_=gv.gvar(0.4, np.inf),show=True):
    nconf_ = ratio1_.coords['jackknife'].size
    t_ = ratio1_.coords['t'].values
    data1_ = ratio1_.sel(t2=t2_).squeeze().transpose('jackknife', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
    data2_ = ratio2_.sel(t2=t2_).squeeze().transpose('jackknife', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
    data1_fit = prep_data_ratio_new_jack(data1_,ncut1_,ncut2_,t2_,t_)
    data2_fit = prep_data_ratio_new_jack(data2_,ncut1_,ncut2_,t2_,t_)
    x_temp={}
    x_temp['1'] = data1_fit[0]
    x_temp['2'] = data2_fit[0]
    data_temp = {}
    data_temp['1'] = data1_fit[1]
    data_temp['2'] = data2_fit[1]
    data_fit = (x_temp,data_temp)

    prior0_2st_3t_combine2['dm'] = dm_ 
    fit_ = lsqfit.nonlinear_fit(data=data_fit, prior=prior0_2st_3t_combine2, fcn=fcn_2st_3t_combine2, debug=True)
    if show:
        print_fit(fit_)
        print(fit_.p['c0_2']-fit_.p['c0_1'])
    return fit_


prior0_2st_3t_combine3_2pt={}
prior0_2st_3t_combine3_2pt['c0_1'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3_2pt['c0_2'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3_2pt['c0_3'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3_2pt['c1_1'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3_2pt['c1_2'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3_2pt['c1_3'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3_2pt['c2_1'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3_2pt['c2_2'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3_2pt['c2_3'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3_2pt['dm'] = gv.gvar(0.3,np.inf) 
prior0_2st_3t_combine3_2pt['A'] = gv.gvar(1,np.inf) 
prior0_2st_3t_combine3_2pt['m0'] = gv.gvar(0.5,np.inf) 
prior0_2st_3t_combine3_2pt['B'] = gv.gvar(1,np.inf) 

def fcn_2st_3t_combine3_2pt(t_, p_):
    c01_ = p_['c0_1']
    c11_ = p_['c1_1']
    c21_ = p_['c2_1']
    c02_ = p_['c0_2']
    c12_ = p_['c1_2']
    c22_ = p_['c2_2']
    c03_ = p_['c0_3']
    c13_ = p_['c1_3']
    c23_ = p_['c2_3']
    dm_ = p_['dm']
    A_ = p_['A']
    m0_ = p_['m0']
    B_ = p_['B']
    ans = {}
    ans['1'] = c01_ + c11_ * np.exp(-dm_ * (t_['1'][:, 1] - t_['1'][:, 0])) + c21_ * np.exp(-dm_ * t_['1'][:, 0])
    ans['2'] = c02_ + c12_ * np.exp(-dm_ * (t_['2'][:, 1] - t_['2'][:, 0])) + c22_ * np.exp(-dm_ * t_['2'][:, 0])
    ans['3'] = c03_ + c13_ * np.exp(-dm_ * (t_['3'][:, 1] - t_['3'][:, 0])) + c23_ * np.exp(-dm_ * t_['3'][:, 0])
    ans['4'] = A_ * np.exp(-m0_* t_['4']) + B_ * np.exp(-(m0_+dm_) * t_['4']) 
    return ans

def do_2st_3t_fit_combine3_2pt(ratio1_,ratio2_, twoptfn_t, twoptfn_gvr, t2_,ncut1_=1,ncut2_=0,dm_=gv.gvar(0.4, np.inf),show=True):
    nconf_ = ratio1_.coords['conf'].size
    t_ = ratio1_.coords['t'].values
    
    #data1_ = ratio1_.sel(t2=t2_).squeeze().transpose('conf', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
    #data2_ = ratio2_.sel(t2=t2_).squeeze().transpose('conf', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
    #data3_ = data2_ - data1_ 
    #data1_fit = prep_data_ratio_new(data1_,ncut1_,ncut2_,t2_,t_)
    #data2_fit = prep_data_ratio_new(data2_,ncut1_,ncut2_,t2_,t_)
    #data3_fit = prep_data_ratio_new(data3_,ncut1_,ncut2_,t2_,t_)
    #x_temp={}
    #x_temp['1'] = data1_fit[0]
    #x_temp['2'] = data2_fit[0]
    #x_temp['3'] = data3_fit[0]
    #data_temp = {}
    #data_temp['1'] = data1_fit[1]
    #data_temp['2'] = data2_fit[1]
    #data_temp['3'] = data3_fit[1]
    
    data_ = xr.concat([ratio1_, ratio2_, ratio2_-ratio1_], dim='operator')
    data_ = data_.squeeze().transpose('conf', 'operator','t2', 't').values.reshape(nconf_, 3 * t2_.size * t_.size) 
    data_ave = np.average(data_, 0)
    data_err = np.std(data_, 0) / np.sqrt(nconf_ - 1)
    data_cov = np.cov(data_, rowvar=False) / (nconf_ - 1) #FIXME
    data_gvr = gv.gvar(data_ave, data_cov).reshape(3, t2_.size, t_.size)
    #data_gvr = gv.gvar(data_ave, data_err).reshape(3, t2_.size, t_.size)
    x_temp={}
    x_temp['1'] = [] 
    x_temp['2'] = [] 
    x_temp['3'] = [] 
    data_temp = {}
    data_temp['1'] = []
    data_temp['2'] = []
    data_temp['3'] = []
    for it2 in range(t2_.size):
        for it in range(ncut1_, t2_[it2] - ncut2_ + 1):
            x_temp['1'].append((it, t2_[it2]))
            x_temp['2'].append((it, t2_[it2]))
            x_temp['3'].append((it, t2_[it2]))
            data_temp['1'].append(data_gvr[0, it2, it])
            data_temp['2'].append(data_gvr[1, it2, it])
            data_temp['3'].append(data_gvr[2, it2, it])
    x_temp['1'] = np.array(x_temp['1']) 
    x_temp['2'] = np.array(x_temp['2']) 
    x_temp['3'] = np.array(x_temp['3']) 
    data_temp['1'] = np.array(data_temp['1']) 
    data_temp['2'] = np.array(data_temp['2'])
    data_temp['3'] = np.array(data_temp['3'])
    
    x_temp['4'] = twoptfn_t
    data_temp['4'] = twoptfn_gvr
    data_fit = (x_temp,data_temp)
    
    fit_ = lsqfit.nonlinear_fit(data=(twoptfn_t[-5:],twoptfn_gvr[-5:]), prior=prior0_exp, fcn=fcn_exp, debug=True)
    prior0_2st_3t_combine3_2pt['m0'] = gv.gvar(fit_.pmean['c1'],fit_.psdev['c1']*5)

    prior0_2st_3t_combine3_2pt['dm'] = dm_ 
    fit_ = lsqfit.nonlinear_fit(data=data_fit, prior=prior0_2st_3t_combine3_2pt, fcn=fcn_2st_3t_combine3_2pt, debug=True, svdcut=1e-6)
    if show:
        print_fit(fit_)
        print(fit_.p['c0_2']-fit_.p['c0_1'])
    return fit_


prior0_2st_3t_combine3={}
prior0_2st_3t_combine3['c0_1'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3['c0_2'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3['c0_3'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3['c1_1'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3['c1_2'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3['c1_3'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3['c2_1'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3['c2_2'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3['c2_3'] = gv.gvar(0.1,np.inf) 
prior0_2st_3t_combine3['dm'] = gv.gvar(0.3,np.inf) 

def fcn_2st_3t_combine3(t_, p_):
    c01_ = p_['c0_1']
    c11_ = p_['c1_1']
    c21_ = p_['c2_1']
    c02_ = p_['c0_2']
    c12_ = p_['c1_2']
    c22_ = p_['c2_2']
    c03_ = p_['c0_3']
    c13_ = p_['c1_3']
    c23_ = p_['c2_3']
    dm_ = p_['dm']
    ans = {}
    ans['1'] = c01_ + c11_ * np.exp(-dm_ * (t_['1'][:, 1] - t_['1'][:, 0])) + c21_ * np.exp(-dm_ * t_['1'][:, 0])
    ans['2'] = c02_ + c12_ * np.exp(-dm_ * (t_['1'][:, 1] - t_['1'][:, 0])) + c22_ * np.exp(-dm_ * t_['1'][:, 0])
    ans['3'] = c03_ + c13_ * np.exp(-dm_ * (t_['1'][:, 1] - t_['1'][:, 0])) + c23_ * np.exp(-dm_ * t_['1'][:, 0])
    return ans

def do_2st_3t_fit_combine3(ratio1_,ratio2_,t2_,ncut1_=1,ncut2_=0,dm_=gv.gvar(0.4, np.inf),show=True):
    nconf_ = ratio1_.coords['conf'].size
    t_ = ratio1_.coords['t'].values
    
    #data1_ = ratio1_.sel(t2=t2_).squeeze().transpose('conf', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
    #data2_ = ratio2_.sel(t2=t2_).squeeze().transpose('conf', 't2', 't').values.reshape(nconf_, t2_.size * t_.size)
    #data3_ = data2_ - data1_ 
    #data1_fit = prep_data_ratio_new(data1_,ncut1_,ncut2_,t2_,t_)
    #data2_fit = prep_data_ratio_new(data2_,ncut1_,ncut2_,t2_,t_)
    #data3_fit = prep_data_ratio_new(data3_,ncut1_,ncut2_,t2_,t_)
    #x_temp={}
    #x_temp['1'] = data1_fit[0]
    #x_temp['2'] = data2_fit[0]
    #x_temp['3'] = data3_fit[0]
    #data_temp = {}
    #data_temp['1'] = data1_fit[1]
    #data_temp['2'] = data2_fit[1]
    #data_temp['3'] = data3_fit[1]
    
    data_ = xr.concat([ratio1_, ratio2_, ratio2_-ratio1_], dim='operator')
    data_ = data_.squeeze().transpose('conf', 'operator','t2', 't').values.reshape(nconf_, 3 * t2_.size * t_.size) 
    data_ave = np.average(data_, 0)
    data_err = np.std(data_, 0) / np.sqrt(nconf_ - 1)
    data_cov = np.cov(data_, rowvar=False) / (nconf_ - 1) #FIXME
    data_gvr = gv.gvar(data_ave, data_cov).reshape(3, t2_.size, t_.size)
    #data_gvr = gv.gvar(data_ave, data_err).reshape(3, t2_.size, t_.size)
    x_temp={}
    x_temp['1'] = [] 
    x_temp['2'] = [] 
    x_temp['3'] = [] 
    data_temp = {}
    data_temp['1'] = []
    data_temp['2'] = []
    data_temp['3'] = []
    for it2 in range(t2_.size):
        for it in range(ncut1_, t2_[it2] - ncut2_ + 1):
            x_temp['1'].append((it, t2_[it2]))
            x_temp['2'].append((it, t2_[it2]))
            x_temp['3'].append((it, t2_[it2]))
            data_temp['1'].append(data_gvr[0, it2, it])
            data_temp['2'].append(data_gvr[1, it2, it])
            data_temp['3'].append(data_gvr[2, it2, it])
    x_temp['1'] = np.array(x_temp['1']) 
    x_temp['2'] = np.array(x_temp['2']) 
    x_temp['3'] = np.array(x_temp['3']) 
    data_temp['1'] = np.array(data_temp['1']) 
    data_temp['2'] = np.array(data_temp['2'])
    data_temp['3'] = np.array(data_temp['3'])
    
    data_fit = (x_temp,data_temp)

    prior0_2st_3t_combine3['dm'] = dm_ 
    fit_ = lsqfit.nonlinear_fit(data=data_fit, prior=prior0_2st_3t_combine3, fcn=fcn_2st_3t_combine3, debug=True, svdcut=1e-8)
    if show:
        print_fit(fit_)
        print(fit_.p['c0_2']-fit_.p['c0_1'])
    return fit_


fmt = ['v','^','d','o','s','p']
clr = ['r','b','k','g','m','purple','orange','c']

def plot_2st(plt_, pdf_, fit_, t2_plot_, ratio_ave_, ratio_err_, ncut1_=1, ncut2_=0, ylabel_='', title_='', save=True, close=True):
    count_ = 0
    plt_.xlabel(r'$t-\frac{t_f}{2}$',fontsize=14)
    if ylabel_ != '':
        plt_.ylabel(ylabel_,fontsize=14)
    for it in range(t2_plot_.size):
        if title_ != '':
            plt_.title(title_)
        plt_.errorbar(np.arange(ncut1_, t2_plot_[it]-ncut2_) - int(t2_plot_[it]/2)+count_*0.05, ratio_ave_.sel(t2=t2_plot_[it])[ncut1_:t2_plot_[it]-ncut2_],
                     ratio_err_.sel(t2=t2_plot_[it])[ncut1_:t2_plot_[it]-ncut2_], label='sep=%d'%(t2_plot_[it]), fmt=fmt[count_], color=clr[count_])
        count_ += 1
    plt_.fill_between(np.arange(ncut1_, t2_plot_[it]-ncut2_) - int(t2_plot_[it]/2)+count_*0.05,
                      fit_.p['c0'].mean-fit_.p['c0'].sdev, fit_.p['c0'].mean+fit_.p['c0'].sdev, alpha=0.5, label='fit')
    plt_.text(plt_.axis()[0] + 0.75 * (plt_.axis()[1] - plt_.axis()[0]), plt_.axis()[2] + 0.05 * (plt_.axis()[3] - plt_.axis()[2]),
              r'$\chi^2$/dof=%4.2f' % (fit_.chi2/fit_.dof))
    plt_.text(plt_.axis()[0] + 0.75 * (plt_.axis()[1] - plt_.axis()[0]), plt_.axis()[2] + 0.10 * (plt_.axis()[3] - plt_.axis()[2]), r'$c_0$=%s' % str(fit_.p['c0']))
    plt_.legend(fontsize=14)
    plt_.tight_layout()
    if save:
        pdf_.savefig()
    if close:
        plt_.close()


def get_DSR(ratio_, sum_d1_=1, sum_d2_=0):
    ratio_sum_ = ratio_.copy()
    for itf in range(16):
        ratio_sum_[dict(t2=itf)][dict(t=0)] = 0.0
        for it in range(sum_d1_, itf-sum_d2_):
            ratio_sum_[dict(t2=itf)][dict(t=0)] += ratio_[dict(t2=itf)][dict(t=it)]

    ratio_sum2_ = ratio_.copy()
    for itf in range(16):
        ratio_sum2_[dict(t2=itf)][dict(t=0)] = 0.0
        for it in range(sum_d1_, itf-sum_d2_-1):
            ratio_sum2_[dict(t2=itf)][dict(t=0)] += ratio_[dict(t2=itf-1)][dict(t=it)]

    ratio_sum_d_ = ratio_sum_ - ratio_sum2_
    ratio_sum_d_ave_ = ratio_sum_d_.lattice.mean()
    ratio_sum_d_err_ = ratio_sum_d_.lattice.std_error()
    return ratio_sum_d_ave_, ratio_sum_d_err_


def get_means_and_errors_gvar(data):
    means_ = np.array([data[i].mean for i in range(data.size)])
    errors_ = np.array([data[i].sdev for i in range(data.size)])
    return means_, errors_


def get_means_gvar(data):
    means_ = np.array([data[i].mean for i in range(data.size)])
    return means_


def get_errors_gvar(data):
    errors_ = np.array([data[i].sdev for i in range(data.size)])
    return errors_


def add_gvar(a_, b_):
    ma_, ea_ = get_means_and_errors_gvar(a_)
    mb_, eb_ = get_means_and_errors_gvar(b_)
    return gv.gvar(ma_+mb_, (ea_**2+eb_**2)**0.5)


def combine_cov_gvar(list_of_gv_vectors):
    v = []
    e = []
    for i in list_of_gv_vectors:
        v += list(get_means_gvar(i))
        e += list(get_errors_gvar(i))
    new = gv.gvar(v,e)
    new = new.reshape(len(list_of_gv_vectors), list_of_gv_vectors[0].size)
    return new

mpi_48I = np.array([
0.113765, 
0.133914,
0.148832,
0.181334,
0.206835,
0.266587,
0.331329,
0.37208
])
mpi_32I = np.array([
0.259627, 
0.294634,
0.315765,
0.353466,
0.409763
])
mpi_24I = np.array([
0.253606, 
0.281943,
0.321064,
0.347715,
0.389302
])
mpi_32ID = np.array([
1.067199660177542e-01, 
1.265659154846626e-01,
1.693844698633821e-01,
1.902826167054158e-01,
2.090192518527308e-01,
2.368994786235278e-01
]) * 1.3784
mpi_32ID250 = np.array([
175.44*1e-3,  
234.20*1e-3,
251.86*1e-3,
262.96*1e-3,
288.73*1e-3,
327.09*1e-3,
392.72*1e-3,])
mpi_32IF = np.array([
0.10928, 
0.12182,    
0.13334,    
0.15420,
0.17294,    
0.22132,    
0.23217    
]) * 3.148

# a m_pi m_pi_sea m_pi*L
x24I = []
x24I.append((0.1105,0.253606,0.337,24/1.7848))
x24I.append((0.1105,0.281943,0.337,24/1.7848))
x24I.append((0.1105,0.321064,0.337,24/1.7848))
x24I.append((0.1105,0.347715,0.337,24/1.7848))
x24I.append((0.1105,0.389302,0.337,24/1.7848))

x32I = []
x32I.append((0.0828,0.259627,0.302,32/2.3833))
x32I.append((0.0828,0.294634,0.302,32/2.3833))
x32I.append((0.0828,0.315765,0.302,32/2.3833))
x32I.append((0.0828,0.353466,0.302,32/2.3833))
x32I.append((0.0828,0.409763,0.302,32/2.3833))

x48I = []
x48I.append((0.1141,0.113765,0.139,48/1.7295))
x48I.append((0.1141,0.133914,0.139,48/1.7295))
x48I.append((0.1141,0.148832,0.139,48/1.7295))
x48I.append((0.1141,0.181334,0.139,48/1.7295))
x48I.append((0.1141,0.206835,0.139,48/1.7295))
x48I.append((0.1141,0.266587,0.139,48/1.7295))
x48I.append((0.1141,0.331329,0.139,48/1.7295))
x48I.append((0.1141,0.372080,0.139,48/1.7295))

x32ID = []
x32ID.append((0.1431,1.067199660177542e-01*1.3784,0.1727,32/1.3784))
x32ID.append((0.1431,1.265659154846626e-01*1.3784,0.1727,32/1.3784))
x32ID.append((0.1431,1.693844698633821e-01*1.3784,0.1727,32/1.3784))
x32ID.append((0.1431,1.902826167054158e-01*1.3784,0.1727,32/1.3784))
x32ID.append((0.1431,2.090192518527308e-01*1.3784,0.1727,32/1.3784))
x32ID.append((0.1431,2.368994786235278e-01*1.3784,0.1727,32/1.3784))

x32ID250 = []
x32ID250.append((0.1431,175.44*1e-3,0.2501,32/1.3784))
x32ID250.append((0.1431,234.20*1e-3,0.2501,32/1.3784))
x32ID250.append((0.1431,251.86*1e-3,0.2501,32/1.3784))
x32ID250.append((0.1431,262.96*1e-3,0.2501,32/1.3784))
x32ID250.append((0.1431,288.73*1e-3,0.2501,32/1.3784))
x32ID250.append((0.1431,327.09*1e-3,0.2501,32/1.3784))
x32ID250.append((0.1431,392.72*1e-3,0.2501,32/1.3784))

x32IF = []
x32IF.append((0.0627,0.10928*3.148,0.371,32/3.148))
x32IF.append((0.0627,0.12182*3.148,0.371,32/3.148))
x32IF.append((0.0627,0.13334*3.148,0.371,32/3.148))
x32IF.append((0.0627,0.15420*3.148,0.371,32/3.148))
x32IF.append((0.0627,0.17294*3.148,0.371,32/3.148))
x32IF.append((0.0627,0.22132*3.148,0.371,32/3.148))
x32IF.append((0.0627,0.23217*3.148,0.371,32/3.148))

p0_global = {}
p0_global['c0'] = -0.1
p0_global['c1'] = 0.1
p0_global['c2'] = 0.1
p0_global['c3'] = 0.1
p0_global['c4'] = 0.1

def fcn_global(x_, p_):
    c0 = p_['c0']
    c1 = p_['c1']
    c2 = p_['c2']
    c3 = p_['c3']
    c4 = p_['c4']
    a = x_[:,0]
    m = x_[:,1]
    ms = x_[:,2]
    L = x_[:,3]
    m0 = 0.138
    #return c0 + c1*x_[:,0]**2 + c2*(x_[:,1]**2-0.14**2) + c3*(x_[:,2]**2-0.14**2) + c4*np.exp(-x_[:,1]*x_[:,3])
    return c0 + c1*(m**2-m0**2) + c2*(m**2-ms**2) + c3*(a**2) + c4*np.exp(-m*L)
