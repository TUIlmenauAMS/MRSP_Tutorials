import numpy as np

def extract_baseband_from_Fa(Fa, orientation='normal'):
    """
    Extracts the baseband prototype h(n) from the analysis folding matrix Fa(z).

    Parameters
    ----------
    Fa : ndarray, shape (N, N, Deg)
        Polynomial matrix tensor of the analysis folding matrix.
        Fa[:,:,m] holds the coefficient matrix of z^(-m).
        Assumes the "diamond" sparsity/layout used for MDCT/LDFB analysis Fa.
    orientation : {'normal','flipped'}, optional
        'normal'   -> standard MDCT/LDFB diamond (as produced by Famatrix).
        'flipped'  -> horizontally flipped diamond (odd number of zero-delay
                      stages may lead to this; if your first try fails, try flipped).

    Returns
    -------
    h : ndarray, shape (L,)
        Reconstructed baseband prototype impulse response, with
        L = Deg * N.
    """
    N0, N1, Deg = Fa.shape
    assert N0 == N1, "Fa must be square in its first two dims."
    N = N0
    half = N // 2

    blocks = []
    for m in range(Deg):
        M = Fa[:, :, m]

        if orientation == 'normal':
            # Invert the exact assignments used in Famatrix(h):
            # Fa[0:half,    0:half   ] = -fliplr(diag(h[0:half]))
            # Fa[half:N,    0:half   ] = -diag(h[half:N])
            # Fa[0:half,    half:N   ] = -diag(h[N:N+half])
            # Fa[half:N,    half:N   ] =  fliplr(diag(h[N+half:2N]))
            q1 = -np.diag(np.fliplr(M[0:half, 0:half]))
            q2 = -np.diag(M[half:N, 0:half])
            q3 = -np.diag(M[0:half, half:N])
            q4 =  np.diag(np.fliplr(M[half:N, half:N]))
        elif orientation == 'flipped':
            # Horizontal flip: swap left/right roles in the extraction.
            q1 = -np.diag(np.fliplr(M[0:half, half:N]))
            q2 = -np.diag(M[half:N,  half:N])
            q3 = -np.diag(M[0:half,  0:half])
            q4 =  np.diag(np.fliplr(M[half:N, 0:half]))
        else:
            raise ValueError("orientation must be 'normal' or 'flipped'")

        # This block corresponds to a reversed h-chunk (Famatrix used h[::-1] internally)
        h_rev_block = np.concatenate([q1, q2, q3, q4])
        blocks.append(h_rev_block)

    # Stack blocks (each length N) and undo the global reverse used in Famatrix:
    h_rev = np.concatenate(blocks)              # length Deg*N
    h = h_rev[::-1].astype(float)               # back to increasing n

    return h


# ---------- tiny self-test ----------
if __name__ == "__main__":
    # Rebuild Fa from a known h using the earlier Famatrix layout, then extract h back.
    def Famatrix(h):
        """Diamond folding matrix (analysis) from h, as in the chapter."""
        N = int(len(h)//2)
        h_rev = h[::-1]
        Fa = np.zeros((N, N, 1), dtype=float)
        half = N//2
        Fa[0:half,  0:half,  0] = -np.fliplr(np.diag(h_rev[0:half]))
        Fa[half:N,  0:half,  0] = -np.diag(h_rev[half:N])
        Fa[0:half,  half:N,  0] = -np.diag(h_rev[N:N+half])
        Fa[half:N,  half:N,  0] =  np.fliplr(np.diag(h_rev[N+half:2*N]))
        return Fa

    # Example prototype of length 2N:
    N = 8
    h_true = np.hanning(2*N)  # any length-2N vector
    Fa = Famatrix(h_true)     # build diamond folding matrix (z^0 only → Deg=1)

    h_est = extract_baseband_from_Fa(Fa, orientation='normal')
    print("max abs error:", np.max(np.abs(h_est - h_true)))

