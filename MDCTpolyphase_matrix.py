#Here is a Python implementation of such a polyphase matrix calculation for the MDCT:
#Started by ChatGPT
#Gerald Schuller, July 2024

import numpy as np

def polyphase_matrix(h, N, ph):
    #h: baseband prototype, N: Number of subbands, ph: phase sign, +1 for analysis, -1 for synthesis 
    L = len(h)
    P = L // N
    H = np.zeros((N, N, P), dtype=complex)
    for p in range(P):
        for k in range(N):
            T0 = np.cos(np.pi / N * (k+0.5) * (p * N + np.arange(N) +ph*N/2 + 0.5))
            H[:, k, p] = h[p * N + np.arange(N)] * T0
        if ph==-1: #Synthesis has time and frequency indices transposed:
            H[:, :, p]=H[:, :, p].transpose()
        else: #Analysis needs flipping in time dimension:
            H[:, :, p]=np.flip(H[:, :, p], axis=0)
    return H
if __name__ == '__main__':
    h = np.sin(np.pi / 8 * (np.arange(8) + 0.5))
    N = 4
    H = polyphase_matrix(h, N, 1)
    print("H[:,:,0]=\n", H[:,:,0])
    print("H[:,:,1]=\n", H[:,:,1])
