from typing import List
import numpy as np
import torch
import bisect
import os,sys
sys.path.append(os.path.dirname(__file__))
from my_lib_colab5 import *
from typing import Union


class OBDemu():
    """
    One body decaying dark matter emulator, as developed and described in 2104.07675 (10.1088/1475-7516/2021/10/040)
    """
    def __init__(self):
        print('One-body decays emulator loaded!')

    def predict(self, k: Union[float,list], z: Union[float,list], Gamma: float = 1e-10, f: float = 0., Ob: float =0.049, Om: float = 0.315, h:float = 0.67):
        """
        Enqvist et al. 2015 calibrated on simulations by Jonathan Hubert & Aurel Schneider
        :param k: scale in h/Mpc
        :param z: redshift
        :param Gamma: decay rate of dark matter in 1/Gyr
        :param f: fraction of decaying dark matter in total dark matter budget
        :param Ob: total matter abundance
        :param Om: total matter abundance
        :param h: present value of hubble constant
        :return: ratio of nonlinear matter power spectra of decaying DM and LCDM
        """

        # convert big omegas (input) to small omegas (used for the fit)
        wb = Ob*h*h
        wm = Om*h*h

        # convert k from h/Mpc to 1/Mpc
        k = k*h

        # Run some checks
        assert f >= 0. and f <= 1., "f is not within (0,1), chosen f is: {}".format(f) # well-defined f
        assert Gamma >= 0, "Gamma is not positive, chosen Gamma is: {}".format(Gamma)
        for i in range(len(k)):
            assert k[i] <= 10 and k[i] >= 1e-6, "found k value not in good range (1e-6,10) [1/Mpc]. Value is {}".format(k[i])
        assert z.all() >= 0, "z is not positive, chosen z is: {}".format(z)

        # These are only to check if the parameter is in a range where low error is expected.
        # Fit can still function well outside this range if the values are not extreme
        if(z.any() > 1): # We know the fitting function is more accurate for lower redshifts < 1
            print("You have chosen a higher redshift z={}!\n-> the fit could be unaccurate with this choice! (error might be > 10%)".format(z))
        if(Gamma >= 0.0316): # We know the fitting function is more accurate for higher lifetimes > 31.6 Gyr
            print("You have chosen a short lifetime of {} Gyr!\n-> the fit could be unaccurate with this choice! (error might be > 10%)".format(Gamma**-1.))
        if(wb< 0.019 or wb > 0.026):
            print("You have chosen wb={}!\n-> the fit could be unaccurate with this choice! (error might be > 10%)".format(wb))
        if(wm< 0.09 or wm > 0.28):
            print("You have chosen wm={}!\n-> the fit could be unaccurate with this choice! (error might be > 10%)".format(wm))
        if(h< 0.6 or h > 0.8):
            print("You have chosen h={}!\n-> the fit could be unaccurate with this choice! (error might be > 10%)".format(h))

        # Fit calculation, notice that we need k in h/Mpc for the calculation (fit is built upon sims with h/Mpc)
        # k = k*h

        a = 0.7208 + 2.027*Gamma + (3.431 - 0.4)*(1./(1.+z*1.1)) - 0.18
        b = 0.0120 + 2.786*Gamma + (0.6499 + 0.02)*(1./(1.+z*1.1)) - 0.09
        p = 1.045 + 1.225*Gamma + (0.2207)*(1./(1.+z*1.1)) - 0.099
        q = 0.9922 + 1.735*Gamma + (0.2154)*(1./(1.+z*1.1)) - 0.056

        u = wb/0.02216
        v = h/0.6776
        w = wm/0.1412
        
        eps1 = 5.323 - 1.4644*u - 1.391*v + (-2.055 +1.329*u + 0.8672*v)*w + (0.2682 - 0.3509*u)*w*w
        eps2 = 0.9260 + (0.05735 - 0.02690*v)*w + (-0.01373 + 0.006713*v)*w*w
        eps3 = (9.553 - 0.7860*v) + (0.4884 + 0.1754*v)*w + (-0.2512 + 0.07558*v)*w*w

        linear_fit = eps1*((Gamma)**eps2)*((1./(1.+z*0.105))**eps3)
        nonlinear_fit = (1.+a*(k**p))/(1.+b*(k**q))*(f)

        # print('k-shape,z-shape,PP-shape',k.shape,z.shape,(-linear_fit*nonlinear_fit + 1.).shape)
        # print('a,b,p,q,eps1,eps2,eps3,linear fit, nonlinear fit:',a.shape,b.shape,p.shape,q.shape,eps1.shape,eps2.shape,eps3.shape,linear_fit.shape,nonlinear_fit.shape)
        # print('z,k:',z[:20],k[:20])
        
        return -linear_fit*nonlinear_fit + 1.

class TBDemu():
    def __init__(self):
        PATH_TO_EMULATOR = os.path.dirname(__file__) + '/files/'
        self.emulator_DDM_2body = torch.load(PATH_TO_EMULATOR + 'emulator3D_bohb_optimal.pth',map_location=torch.device('cpu'))
        self.t_means = torch.tensor(np.genfromtxt(PATH_TO_EMULATOR+'PCA_means.csv'))
        self.t_eigenvectors = torch.tensor(np.genfromtxt(PATH_TO_EMULATOR+'PCA_eigenvectors.csv'))

    def check_interpret_k_z_input(self,k: Union[float,list], z: Union[float,list]):
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
            
            
        

    def predict(self, k: Union[float,list], z: Union[float,list], f: float = 1.0, vk: float = 0., Gamma: float = 1e-10, p: float = 0.0) -> list:
        
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
