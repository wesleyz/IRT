import pystan
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import utils


class IRT4Order:
    def __init__(self, df: pd.core.frame.DataFrame,
                 nc: int, stm=None, dat=None, D=1):
        self.df = df
        self.nj = df.shape[1]
        self.ni = df.shape[0]
        self.nc = nc
        self.D = D

        self.fit = None
        self.result = None
        self.alpha = None
        self.beta = None
        self.theta = None

    def set_model(self, sample=None):
        self.stm = pystan.StanModel(model_code=utils.model)
        if sample is not None:
            self.ni = sample

        self.dat = {"y": self.df.sample(
            self.ni).values, "nj": self.nj, "ni": self.ni, "nc": self.nc,
            "D": self.D}

    def model_fit(self, n_itr=2000, chains=4, n_warmup=1000, n_jpbs=-1,
                  algorithm="NUTS", verbose=False):
        self.fit = self.stm.sampling(data=self.dat, iter=n_itr,
                                     chains=chains, n_jobs=-1,
                                     warmup=n_warmup, algorithm="NUTS",
                                     verbose=False)

    def extract(self):
        self.result = self.fit.extract()
        self.alpha = self.result["a"].mean(axis=0)
        self.beta = self.result["b"].mean(axis=0)
        self.theta = self.result["theta"].mean(axis=0)

    def plot(self, sort="Q", save_name="irt_Q"):
        theta_ = np.arange(np.min(self.theta), np.max(self.theta),
                           step=(np.max(self.theta)
                                 - np.min(self.theta))/self.ni)
        irt_plot = np.array([[1/(1+np.exp(-self.alpha[j]*(theta_
                                                          - self.beta[j, c])))
                              for c in range(self.nc)]
                             for j in range(self.nj)])
        if sort == "Q":
            for j in range(self.nj):
                for c in range(self.nc):
                    plt.title("irt_plot_Q{q_num}".format(q_num=j))
                    plt.plot(irt_plot[j][c], label=str(c))
                    plt.xticks(np.arange(0, self.ni, step=10),
                               np.round(np.arange(np.min(self.theta),
                                                  np.max(self.theta),
                                                  step=(np.max(self.theta)
                                                        - np.min(self.theta))
                                                  / 10),
                                        decimals=2))
                    plt.legend()
                    plt.savefig(save_name+str(j))
                plt.show()

        elif sort == "L":
            for c in range(self.nc):
                for j in range(self.nj):
                    plt.title("irt_plot_Level{level}".format(level=c))
                    plt.plot(irt_plot[j][c], label="Q"+str(j))
                    plt.xticks(np.arange(0, self.ni, step=10),
                               np.round(np.arange(np.min(self.theta),
                                                  np.max(self.theta),
                                                  step=(np.max(self.theta)
                                                        - np.min(self.theta))
                                                  / 10),
                                        decimals=2))
                    plt.legend()
                    plt.savefig(save_name+str(c))
                plt.show()

        else:
            print("sort must be Q or L")
