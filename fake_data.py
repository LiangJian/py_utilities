import numpy as np


class FakeData:
    Nt = 0
    off = 0
    m = np.empty(0)
    A = np.empty(0)
    vector_t = np.empty(0)
    vector_D = np.empty(0)
    vector_E = np.empty(0)
    snr = 1.

    @classmethod
    def __init__(cls, nt_=4, off_=1, m_=np.zeros(1), a_=np.zeros(1), snr_=1., seed_=0):
        assert m_.size == a_.size
        cls.Nt = nt_
        cls.off = off_
        cls.m = m_
        cls.A = a_

        np.random.seed(seed_)

        cls.vector_t = np.arange(cls.Nt) + cls.off
        vector_c = np.zeros(cls.Nt)
        for i in range(0, len(cls.m)):
            vector_c += cls.A[i] * np.exp(-cls.m[i] * cls.vector_t)

        cls.snr = snr_
        vector_n = np.random.normal(0, np.sum(cls.A * np.exp(-cls.m * cls.off)) / cls.snr, cls.Nt)
        m_noise = cls.m[0]
        vector_n = vector_n * np.exp(-m_noise * cls.vector_t)
        cls.vector_D = vector_c + vector_n
        cls.vector_E = np.sum(cls.A * np.exp(-cls.m * cls.off)) / cls.snr * np.exp(-m_noise * cls.vector_t)
