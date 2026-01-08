import numpy as np
from tifffile import *
import gc
from tqdm import tqdm

def extract_TC(ica_matrix_path:str, U_path:str, SVT_path:str, save_path:str, nframes_per_chunk:int)->None:

    ## SVT should be loaded as memmap
    ## When this is indexed, then the part of the data is dynamically loaded to the memory
    SVT = np.load(SVT_path, mmap_mode='r')
    # print(f"SVT.shape:{SVT.shape}")
    nframes_SVT = SVT.shape[1]
    ncomponents_SVT = SVT.shape[0]

    ## U is loaded as the shape of (H, W, Nframes)
    ## This need to be reshaped into (H*W, Nframes)
    U = np.load(U_path)
    U = U.transpose(2, 0, 1).reshape(ncomponents_SVT, -1).T

    ## ica_matrix is loaded with the shape of (n_ICs, H*W)
    ica_matrix = np.load(ica_matrix_path)
    ncomponents_ica = ica_matrix.shape[0]

    idx = np.arange(0,nframes_SVT,nframes_per_chunk,dtype='int')
    if not idx[-1] == nframes_SVT:
        idx = np.hstack([idx,nframes_SVT-1])
    

    tcs = np.empty((ncomponents_ica, nframes_SVT))
    print(f"tcs.shape:{tcs.shape}")    

    # データを処理し、結果を順次書き込む
    # print(f"idx:{idx}")
    for i in tqdm(range(len(idx)-1),desc='Computing TCs from U, SVT, ICs'):
        # print(idx[i], idx[i+1])

        reconst_svd_partial = np.matmul(U, SVT[:, idx[i]:idx[i+1]])
        # print(f"reconst_svd_partial.shape:{reconst_svd_partial.shape}")
        tcs_partial = np.matmul(ica_matrix, reconst_svd_partial)
        # print(f"tcs_partial.shape:{tcs_partial.shape}")

        del reconst_svd_partial
        gc.collect()
        # tcs = np.concatenate([tcs, tcs_partial], axis=1)
        # print(tcs.shape)
        tcs[:, idx[i]:idx[i+1]] = tcs_partial
        # print(f"______________________________")


    np.save(save_path, tcs)

