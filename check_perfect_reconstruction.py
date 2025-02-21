import numpy as np
from MDCTpolyphase_matrix import *

def check_perfect_reconstruction(H, G):
    N = H.shape[0]
    P = H.shape[2]
    
    poly_conv=np.zeros((N,N,2 * P - 1), dtype=complex)
    for p in range(P):
        for q in range(P):
            poly_conv[:,:,p + q]+=np.matmul(H[:, :, p], G[:, :, q])
    print(poly_conv[:,:,1])
    PRmatrix=np.zeros((N,N,2 * P - 1), dtype=complex)
    PRmatrix[:,:,1]=-N/2*np.eye(N) #ideal matrix for perfect reconstrcution with MDCT
    if not np.allclose(poly_conv, PRmatrix):
        return False
    
    return True

# Create the analysis polyphase matrix
N = 4 #Number of subbands
#baseband prototype:
h = np.sin(np.pi / (2*N) * (np.arange(2*N) + 0.5))

H = polyphase_matrix(h, N, 1)
# Create the synthesis polyphase matrix
G = polyphase_matrix(h, N, -1)

# Check perfect reconstruction
is_perfect = check_perfect_reconstruction(H, G)
print("Perfect reconstruction:", is_perfect)
