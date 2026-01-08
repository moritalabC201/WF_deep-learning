from wfield.utils import *
from wfield.imutils import mask_to_3d
from skimage.util import view_as_blocks
from numpy.linalg import svd
import numpy as np
from tqdm import tqdm
from tifffile import *

def bin_stack(

    input_stack:np.ndarray, final:tuple):

    row_original, col_original = input_stack.shape[1:]
    row_final, col_final = final

    bin_row = int(row_original/row_final)
    bin_col = int(col_original/col_final)

    ## reshape the dimention of the original data and remove undesirable axis.
    ## This redundant scripts should retain.
    ## If this code is written in oneliner, it somehow returns erorr.

    cube = np.squeeze(view_as_blocks(input_stack, (1, bin_row, bin_col)))
    cube = np.squeeze(cube)

    ## bin tha data
    bin_stack = np.mean(cube, axis = (3,4))

    return bin_stack
    ## Do not del bindata, 
    ## otherwise returned value will not be available in the following analysis

# this function is added by me
from tifffile import *
def load_tif_block(block, #(filename,onset,number of frame)
                   shape = None,
                   dtype='uint16'): 
    
    '''Loads a block from a binary file (nchannels,W,H)'''
    fname,offset,bsize = block
    nchans,W,H = shape
    framesize = int(nchans*W*H)
    dt = np.dtype(dtype)
    offset = int(offset*2)
    
    # with open(fname,'rb') as fd:
    #     fd.seek(offset*framesize*int(dt.itemsize))
    #     buf = np.fromfile(fd,dtype = dt, count=framesize*bsize)
    # buf = buf.reshape((-1,nchans,W,H),
    #                            order='C')
    # return buf

    with TiffFile(fname) as tif:
        buf = np.empty((2*bsize, W, H))
        #buf = np.empty((bsize, W, H))
        #print(f"buf.shape:{buf.shape}")
        #print(f"(nchans, H, W):{(nchans, W, H)}")
        #print(f"type(tif.pages):{type(tif.pages)}")
        for index, page in enumerate(tif.pages[offset:offset+2*bsize]):
            #print(f"index:{index}")
            buf[index, :, :] = page.asarray()
        buf = buf.reshape((bsize, nchans, W, H),order="c")
        #buf = buf.reshape((2*bsize, nchans, W, H),order="c")

    # tif.close()
    # return buf
    return buf

def approximate_svd_tif(dat, frames_average,
                    onsets = None,
                    k=200, 
                    mask = None,
                    nframes_per_bin = 15,
                    nbinned_frames = 5000,
                    nframes_per_chunk = 500,
                    divide_by_average = True):
    '''
    Approximate single value decomposition by estimating U from the average movie and using it to compute S.VT.
    This is similar to what described in Steinmetz et al. 2017

    Joao Couto - March 2020
    
    This computes the mean centered SVD of the dataset, it does not compute F-F0/F0 a.k.a. df/f.
    Compute it after using the SVD components.

    Inputs:
        dat (array)              : (NFRAMES, NCHANNEL, H, W) 
        k (int)                  : number of components to estimate (200)
        nframes_per_bin (int)    : number of frames to estimate the initial U components
        nbinned_frames (int)     : maximum number frames to estimate the initial U components
        nframes_per_chunk (int)  : window size to load to memory each time.
        divide_by_average (bool) : True   
    Returns:
        U   (array)             : 
        SVT (array)             : 
    '''
    from sklearn.preprocessing import normalize

    ## ここのdat_pathとdimsの設定は旨く行っている
    if hasattr(dat,'filename'):
        dat_path = dat.filename
    else:
        dat_path = None
    dims = dat.shape[1:]

    # the number of bins needs to be larger than k because of the number of components.
    if nbinned_frames < k:
        nframes_per_bin = np.clip(int(np.floor(len(dat)/(k))),1,nframes_per_bin)

    nbinned_frames = np.min([nbinned_frames,
                             int(np.floor(len(dat)/nframes_per_bin))])
    
    idx = np.arange(0,nbinned_frames*nframes_per_bin,nframes_per_bin,
                    dtype='int')
    if not idx[-1] == nbinned_frames*nframes_per_bin:
        idx = np.hstack([idx,nbinned_frames*nframes_per_bin-1])
    binned = np.zeros([len(idx)-1,*dat.shape[1:]],dtype = 'float32')
    for i in tqdm(range(len(idx)-1),desc='Binning raw data'):
        if dat_path is None:
            blk = dat[idx[i]:idx[i+1]] # work when data are loaded to memory
        else:
            blk = load_tif_block((dat_path,idx[i],nframes_per_bin),
                                    shape=dims)
        avg = get_trial_baseline(idx[i],frames_average,onsets)
        if divide_by_average:
            binned[i] = np.mean((blk-(avg + np.float32(1e-5)))
                                / (avg + np.float32(1e-5)), axis=0)
        else:
            binned[i] = np.mean(blk-(avg + np.float32(1e-5)), axis=0)
        if not mask is None:
            b = binned[i]
            mmask = mask_to_3d(mask,[1,*mask.shape])
            b[mmask == 0] = 0

        del blk

    ## ここの出力が異なる
    binned = binned[:, 0, :, :]  # 0チャンネル目だけを選択
    binned = binned.reshape((-1,np.multiply(*dims[-2:])))

    # Get U from the single value decomposition 
    cov = np.dot(binned,binned.T)/binned.shape[1]
    cov = cov.astype('float32')

    u,s,v = svd(cov)
    U = normalize(np.dot(u[:,:k].T, binned),norm='l2',axis=1)
    k = U.shape[0]     # in case the k was smaller (low var)
    # if trials are defined, then use them to chunk data so that the baseline is correct
    if onsets is None:
        idx = np.arange(0,len(dat),nframes_per_chunk,dtype='int')
    else:
        idx = onsets
    if not idx[-1] == len(dat):
        idx = np.hstack([idx,len(dat)-1])
    V = np.zeros((k,*dat.shape[:2]),dtype='float32')
    # Compute SVT
    for i in tqdm(range(len(idx)-1),desc='Computing SVT from the raw data'):
        if dat_path is None:
            blk = dat[idx[i]:idx[i+1]] # work when data are loaded to memory
        else:
            blk = load_tif_block((dat_path,idx[i],idx[i+1]-idx[i]),
                                shape=dims).astype('float32')
        avg = get_trial_baseline(idx[i],frames_average,onsets).astype('float32')
        blk = blk-(avg+np.float32(1e-5))
        if divide_by_average:
            blk /= avg+np.float32(1e-5)
        V[:,idx[i]:idx[i+1],:] = np.dot(
            U, blk.reshape([-1,np.multiply(*dims[1:])]).T).reshape((k,-1,dat.shape[1]))
        del blk

    SVT = V.reshape((k,-1))
    U = U.T.reshape([*dims[-2:],-1])
    return U, SVT
