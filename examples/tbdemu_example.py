import numpy as np
import TBDemu
import matplotlib.pyplot as plt

# load emulator
emul = TBDemu.emulator()

# predict suppressions between kmin and kmax for a single redshift
kmin = 1e-3 # in h/Mpc
kmax = 5 # in h/Mpc
ks = np.logspace(np.log10(kmin),np.log10(kmax),200)
zs = 0.0
velocity_kick = 500 # in km/s
gamma_decay = 1/50 # in 1/Gyr
fraction = 1.0

pks = emul.predict(ks,zs,fraction,velocity_kick,gamma_decay)

# plot
plt.semilogx(ks,pks)
plt.xlabel(r'$k$ [h/Mpc]')
plt.ylabel(r'$P_{\rm DDM}/P_{\Lambda \rm CDM}$')
plt.tight_layout()
plt.show()