from typing import List
import numpy as np
import torch
import bisect
import os,sys
sys.path.append(os.path.dirname(__file__))
from my_lib_colab5 import *

class emulator():
    def __init__(self):
        PATH_TO_EMULATOR = os.path.dirname(__file__) + '/files/'
        self.emulator_DDM_2body = torch.load(PATH_TO_EMULATOR + 'emulator3D_bohb_optimal.pth',map_location=torch.device('cpu'))
        self.t_means = torch.tensor(np.genfromtxt(PATH_TO_EMULATOR+'PCA_means.csv'))
        self.t_eigenvectors = torch.tensor(np.genfromtxt(PATH_TO_EMULATOR+'PCA_eigenvectors.csv'))

    def check_interpret_k_z_input(self,k: float|list, z: float|list):
        # print('k,z types:',type(z) ,type(k))
        single_redshift = False
        if type(k) == float and type(z) == float:
            k = np.array([k])
            z = np.array([z])
            single_redshift = True
        elif type(z) == float and type(k) != float:
            if type(k) == list:
                k = np.array(k)
            z = np.array([z])
            single_redshift = True
        elif type(k) == float and type(z) != float:
            if type(z) == list:
                z = np.array(z)
            k_val = k
            k = np.array([k_val for _ in range(len(z))])
        elif type(k) != float and type(z) != float:
            if len(k) != len(z):
                raise Exception("Dimensions of k and z do not match:",len(k),len(z))
        return k,z,single_redshift
            
            
        

    def predict(self, k: float|list, z: float|list, f: float = 1.0, vk: float = 0., Gamma: float = 1e-10, p: float = 0.0) -> list:
        
        k,z,single_redshift = self.check_interpret_k_z_input(k,z)
        
        #defaults, do not change
        kmin = 1e-3
        kmax = 5.9
        nsteps_emul = 300
        # k_int = np.logspace(np.log10(kmin),np.log10(kmax),nsteps_emul)
        if max(z)>2.35: raise Exception("Too high redshift encountered (>2.35). Aborting...")
        if vk<0 or vk>5000: raise Exception("Velocity kick outside the training domain [0-5000] km/s. Aborting...")
        if Gamma<0 or Gamma>1/13.5: raise Exception("Decay rate outside the training domain [0-1/13.5] 1/Gyr. Aborting...")
        if f<0.0 or f>1.0: raise Exception("Fraction of 2bDDM outside the training domain [0-1]. Aborting...")

        
        kk = 105
        delta = np.log10(kmax/kmin)/(300-1)
        N = int(1+ np.log10(kk/kmin)/delta)
        k_ext = np.logspace(np.log10(kmin),np.log10(kmin) + (N-1)*delta,N)

        
        PP = []

        par_extend = np.tile(np.array([f,transform_v(vk),transform_g(Gamma)]),(len(z),1))
        params_concat = np.concatenate((par_extend,transform_z(z).reshape(-1,1)),axis = 1)

        
        with torch.no_grad():
            model_out_int_nPca, _ = self.emulator_DDM_2body(torch.tensor(params_concat))
        

        Pks_int = self.t_means + torch.matmul(model_out_int_nPca,self.t_eigenvectors)
        Pks_int = torch.clamp(Pks_int,0.,1.).numpy()

        Pks_ext = np.zeros((len(z),N))
        Pks_ext[:,:nsteps_emul] = Pks_int
        # print('Pk_int.shape,Pk_ext.shape:',Pks_int.shape,Pks_ext.shape)
        """
        p: value
        """
        column = Pks_int[:,-1]+(1-Pks_int[:,-1])*p
        extenstion = np.tile(column.reshape(-1,1),(1,N - nsteps_emul))
        Pks_ext[:,nsteps_emul:] = extenstion

        # for tracking of the extrapolation warning
        warning = False
        for i in range(len(k)):
            if k[i]<= kmin: PP.append(1.0) # for very large scales, boost is always 1.0
            else:
                z_idx = 0 if single_redshift == True else i # only one row in Pks_int and Pks_ext
                if k[i]>kmax: 
                    PP.append(column[z_idx])
                    if not warning: warning = True # extrapolation warning will be thrown before return
                else:
                    idx = bisect.bisect_left(k_ext,k[i])
                    lam = np.log10(k[i]/k_ext[idx-1])/np.log10(k_ext[idx]/k_ext[idx-1])
                    Pk_fill = (1-lam)*Pks_ext[z_idx,idx-1] + lam*Pks_ext[z_idx,idx]
                    PP.append(Pk_fill)
        if warning: print('[emulator.predict] Warning: Extrapolation required! Make sure you know what is hapenning.')
        return np.array(PP)


def transform_v(v: float) -> float:
    vks = [0,200,500,1000,5000]
    tvks = np.log10(np.array(vks)+50)
    tvmin = tvks[0]
    tvks -= tvmin
    tvmax = tvks[-1]
    return (np.log10(v + 50) - tvmin)/tvmax


def transform_g(g: float,a: float =115.396) -> float:
    # return g
    return np.log10(a*g+1)


def transform_z(z: float,a: float =3.97) -> float:
    # return z/zs[0]
    return np.log10(a*z+1)


def check_suppression(line: list) -> list:
    for i in range(len(line)):
        if line[i]>1.00:
            line[i] = 1.00
    return line 
