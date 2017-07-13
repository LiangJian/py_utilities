import numpy as np


def mass_eff_cosh(src, index_t):
    return np.arccosh((np.roll(src, +1, index_t)+np.roll(src, -1, index_t))/(2.*src))


def mass_eff_log(src, index_t):
    return np.log(np.roll(src, +1, index_t)/src)


def get_std_error(src, index_conf):
    return np.std(src, index_conf)/np.sqrt(src.shape[index_conf]-1.)


def do_jack(src, index_conf):
    tmp = np.sum(src, index_conf, keepdims=True)
    return (tmp-src)/(src.shape[index_conf]-1)


def get_jack_error(src, index_conf):
    return np.std(src, index_conf) * np.sqrt(src.shape[index_conf]-1)


def get_boot_sample(src):
    if len(src.shape) != 1:
        print("need 1D array")
        exit(-1)
    return np.random.choice(src, src.size)


def get_boot_error(src, index_conf):
    return np.std(src, index_conf)
