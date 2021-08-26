import numpy as np
import matplotlib.pyplot as plt


data = np.load("spectrums/8210.npy")
S11_0 = data[0] + data[1]*1j
S11_90 = data[2] + data[3]*1j
S21_0 = data[4] + data[5]*1j
S21_90 = data[6] + data[7]*1j

plt.figure()
plt.plot(np.abs(S21_0) ** 2, 'r-', label='tran0')
plt.plot(np.abs(S11_0) ** 2, 'b-', label='refl0')
plt.plot(np.abs(S21_90) ** 2, 'r--', label='tran90')
plt.plot(np.abs(S11_90) ** 2, 'b--', label='refl90')
plt.xlabel("Freq (THz)")
plt.ylabel("Transmittance/Reflectance (-)")
plt.legend()
plt.savefig('amp.png')
plt.close()

plt.figure()
plt.plot(np.angle(S21_0), 'r-', label='tran0')
plt.plot(np.angle(S11_0), 'b-', label='refl0')
plt.plot(np.angle(S21_90), 'r--', label='tran90')
plt.plot(np.angle(S11_90), 'b--', label='refl90')
plt.xlabel("Freq (THz)")
plt.ylabel("Transmittance/Reflectance (-)")
plt.legend()
plt.savefig('angle.png')
plt.close()
