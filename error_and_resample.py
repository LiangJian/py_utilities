import numpy as np


def mass_eff_cosh(src, index_t):
    return np.arccosh((np.roll(src, +1, index_t)+np.roll(src, -1, index_t))/(2.*src))


def mass_eff_log(src, index_t):
    return np.log(np.roll(src, +1, index_t)/src)


def get_std_error(src, index_conf):
    return np.std(src, index_conf)/np.sqrt(src.shape[index_conf]-1.)


def do_jack(src, index_conf):
    tmp = np.sum(src, index_conf, keepdims=True)
    return (tmp - src) / (src.shape[index_conf] - 1)


def do_anti_jack(src, index_conf):
    tmp = np.sum(src, index_conf, keepdims=True)
    return tmp - src * (src.shape[index_conf] - 1)


def get_jack_error(src, index_conf):
    return np.std(src, index_conf) * np.sqrt(src.shape[index_conf]-1)


def get_boot_sample(src):
    if len(src.shape) != 1:
        print("need 1D array")
        exit(-1)
    return np.random.choice(src, src.size)


def get_boot_error(src, index_boot):
    return np.std(src, index_boot)


def get_p68_mean_and_error(src):
    # TODO: multi-dimensional case
    # FIXME
    pos_min = int(0.158655 * src.size)
    pos_max = int(0.841345 * src.size)
    v_tmp = np.zeros(pos_max - pos_min)
    vi = np.sort(src)
    for i in range(pos_min, pos_max):
        v_tmp[i - pos_min] = vi[i]
    return np.mean(v_tmp), (vi[pos_max] - vi[pos_min])/2.
