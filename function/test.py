import numpy as np
er=3.3
E0 = 8.8542*1e-12
d=30*1e-6
#A=(0.1791-0.0008)*1e-4
f=337*1e6
#Cap = E0*er*A/d
L = 9*1e-11
#f=1/(2*np.pi*np.sqrt(L*Cap))
Cap = (1/(2*np.pi*f))**2/L
A = Cap*d/E0/er


print(Cap)
print(A*1e4)