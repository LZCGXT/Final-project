from preprocess import load_csi, diff_phase_matrix, get_mValue
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

phase, time, amplitude = load_csi('3_12_after/left/csi/data_0.csi')

phase_matrix, amp_matrix = diff_phase_matrix(phase, time, amplitude)
path = 'gesture_recognition/img/'
plt.subplot(211)
plt.ylabel('amplitude')
plt.xticks([])
plt.plot(time, amplitude.transpose()[0][0][0])

plt.subplot(212)
plt.ylabel('phase')
plt.xlabel('time')
plt.plot(time, phase.transpose()[0][0][0])
impath=path + 'data.jpeg'
plt.savefig(impath, dpi=600)
plt.show()
'''
plt.subplot(211)
#plt.xlabel('time')
plt.ylabel('amplitude')
plt.xticks([])
for i in range(3):
    for j in range(2):
        plt.plot(time, amplitude.transpose()[2][i][j])

b, a = signal.butter(8, 0.7, 'lowpass',analog=False)
n = amplitude.shape[0]
#print(n)
plt.subplot(212)
plt.ylabel('amplitude')
plt.xlabel('time')
filt = np.empty((10, n))
for i in range(3):
    for j in range(2):
        temp = signal.filtfilt(b, a, amplitude.transpose()[2][i][j])
        filt[i] = temp
        plt.plot(time, filt[i])
impath = path + 'butter.jpeg'
plt.savefig(impath, dpi=600)
plt.show()
'''