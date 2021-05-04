from skvideo.utils.mscn import gen_gauss_window, compute_image_mscn_transform
from block import blockMotion
from skvideo.measure.niqe import extract_on_patches
import numpy as np
import scipy.ndimage
import scipy.fftpack
import scipy.stats
import scipy.io
import skvideo.measure as _
import cv2
from numba import jit

niqe_data_path = _.__file__.replace('__init__.py','data/frames_modelparameters.mat')
params = scipy.io.loadmat(niqe_data_path)
mu_prisparam = np.ravel(params["mu_prisparam"]).T
cov_prisparam = params["cov_prisparam"]

@jit(nopython=True)
def eigen_calculation(shape1, shape2, upper_left, off_diag, lower_right):
    Eigens = np.zeros((shape1, shape2, 2), dtype=np.float32)
    for y in range(shape1):
        for x in range(shape2):
            mat = np.array([
                    [upper_left[y, x], off_diag[y, x]],
                    [off_diag[y, x], lower_right[y, x]],])
            w, _ = np.linalg.eig(mat)
            Eigens[y, x] = w
    return Eigens

@jit(nopython=True)
def gamma_calculation(dct_diff5x5, r, mblock, g):
    gamma_matrix = np.zeros((1, mblock**2), dtype=np.float32)
    for s in range(mblock**2):
        temp = dct_diff5x5[:, s]
        mean_gauss = np.mean(temp)
        var_gauss = np.sum((temp - mean_gauss)**2)/(len(temp)-1)
        mean_abs = np.mean(np.abs(temp - mean_gauss))**2
        rho = var_gauss/(mean_abs + 1e-7)

        gamma_gauss = 11
        for x in range(len(g)-1):
          if (rho <= r[x]) and (rho > r[x+1]):
            gamma_gauss = g[x]
            break
        gamma_matrix[0, s] = gamma_gauss
    
    return gamma_matrix

@jit(nopython=True)
def frame_difference(frames, yrange, xrange, mblock):
    diff = np.zeros((xrange*yrange, mblock, mblock), dtype=np.float32)
    count = 0
    for y in range(yrange):
        for x in range(xrange):
            diff[count,:,:] = frames[1, y*mblock:(y+1)*mblock, x*mblock:(x+1)*mblock].astype(np.float32) - frames[0, y*mblock:(y+1)*mblock, x*mblock:(x+1)*mblock].astype(np.float32)
            count += 1
    
    return diff

@jit(nopython=True)
def motion_compensated_frame_difference(frames, motion_vectors,mblock):
    diff_patch = np.zeros((motion_vectors.shape[1]*motion_vectors.shape[2], mblock, mblock), dtype=np.float32)
    count = 0
    for y in range(motion_vectors.shape[1]):
        for x in range(motion_vectors.shape[2]):
            patchP = frames[1, y*mblock:(y+1)*mblock, x*mblock:(x+1)*mblock].astype(np.float32)
            patchI = frames[0, y*mblock+motion_vectors[0, y, x, 0]:(y+1)*mblock+motion_vectors[0, y, x, 0], x*mblock+motion_vectors[0, y, x, 1]:(x+1)*mblock+motion_vectors[0, y, x, 1]].astype(np.float32)
            diff = patchP - patchI
            diff_patch[count,:,:] = diff
            count += 1
    
    return diff_patch

def compute_niqe_features(img):
    # This code snippet is obtained from skvideo.measure.niqe package
    blocksize = 96
    
    h, w = img.shape

    if (h < blocksize) or (w < blocksize):
        print("Input frame is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % blocksize)
    woffset = (w % blocksize)

    if hoffset > 0: 
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    img2 = cv2.resize(img, (w//2,h//2),\
                             interpolation = cv2.INTER_AREA)
    
    mscn1, var, mu = compute_image_mscn_transform(img, extend_mode='nearest')
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2, extend_mode='nearest')
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, blocksize)
    feats_lvl2 = extract_on_patches(mscn2, blocksize//2)
    
    # stack the scale features
    feats = np.hstack((feats_lvl1, feats_lvl2))# feats_lvl3))
    
    #calculate score
    mu_distparam = np.mean(feats, axis=0)
    cov_distparam = np.cov(feats.T)

    invcov_param = np.linalg.pinv((cov_prisparam + cov_distparam)/2, hermitian=True,\
                                  rcond=1e-5)

    xd = mu_prisparam - mu_distparam
    quality = np.sqrt(np.dot(np.dot(xd, invcov_param), xd))

    return np.hstack((mu_distparam, [quality]))

def motion_feature_extraction_frame(frames):
    # setup
    frames = frames.astype(np.float32)
    mblock=10
    h = gen_gauss_window(2, 0.5)
    # step 1: motion vector calculation
    motion_vectors = blockMotion(frames, method='N3SS', mbSize=mblock, p=np.int(1.5*mblock))
    motion_vectors = motion_vectors.astype(np.float32)

    # step 2: compute coherency
    
    motion_frame = motion_vectors[0]

    upper_left = np.zeros_like(motion_frame[:, :, 0])
    lower_right= np.zeros_like(motion_frame[:, :, 0])
    off_diag = np.zeros_like(motion_frame[:, :, 0])
    scipy.ndimage.correlate1d(motion_frame[:, :, 0]**2, h, 0, upper_left, mode='reflect') 
    scipy.ndimage.correlate1d(upper_left, h, 1, upper_left, mode='reflect') 
    scipy.ndimage.correlate1d(motion_frame[:, :, 1]**2, h, 0, lower_right, mode='reflect') 
    scipy.ndimage.correlate1d(lower_right, h, 1, lower_right, mode='reflect') 
    scipy.ndimage.correlate1d(motion_frame[:, :, 1]*motion_frame[:, :, 0], h, 0, off_diag, mode='reflect') 
    scipy.ndimage.correlate1d(off_diag, h, 1, off_diag, mode='reflect')

    Eigens = eigen_calculation(motion_vectors.shape[1],motion_vectors.shape[2],upper_left, off_diag, lower_right) 

    num = (Eigens[:, :, 0] - Eigens[:, :, 1])**2
    den = (Eigens[:, :, 0] + Eigens[:, :, 1])**2

    Coh10x10 = np.zeros_like(num)
    Coh10x10[den!=0] = num[den!=0] / den[den!=0]

    # step 3: global motion
    motion_frame = motion_vectors[0]
    motion_amplitude = np.sqrt(motion_vectors[0, :, :, 0]**2 + motion_vectors[0, :, :, 1]**2) 
    mode10x10 = scipy.stats.mode(motion_amplitude, axis=None)[0][0]
    mean10x10 = np.mean(motion_amplitude)

    motion_diff = np.abs(mode10x10 - mean10x10)

    return Coh10x10, motion_diff, mode10x10

def temporal_dc_variation_feature_extraction_frame(frames):
    frames = frames.astype(np.float32)
    mblock=16
    
    # step 1: motion vector calculation
    motion_vectors = blockMotion(frames, method='N3SS', mbSize=mblock, p=7)
    
    # step 2: compensated temporal dct differences
    diff_patch = motion_compensated_frame_difference(frames, motion_vectors,mblock)
    t = scipy.fftpack.dct(scipy.fftpack.dct(np.array(diff_patch), axis=2, norm='ortho'), axis=1, norm='ortho')
    dct_motion_comp_diff = t[:,0,0]
    std_dc = np.std(dct_motion_comp_diff)
    
    return std_dc

def NSS_spectral_ratios_feature_extraction_frame(frames):
    
    def zigzag(data):
      nrows, ncols = data.shape
      d=sum([list(data[::-1,:].diagonal(i)[::(i+nrows+1)%2*-2+1])for i in range(-nrows,nrows+len(data[0]))], [])
      return np.array(d)

    mblock=5

    # step 1: compute local dct frame differences
    dct_diff5x5 = np.zeros((1,np.int(frames.shape[1]/mblock), np.int(frames.shape[2]/mblock),mblock**2), dtype=np.float32)
    diff_patch = frame_difference(frames, dct_diff5x5.shape[1], dct_diff5x5.shape[2], mblock)
    
    t = scipy.fftpack.dct(scipy.fftpack.dct(diff_patch, axis=2, norm='ortho'), axis=1, norm='ortho')
    dct_diff5x5 = t.reshape(t.shape[0], -1)

    # step 2: compute gamma
    g = np.arange(0.03, 10+0.001, 0.001)
    r = (scipy.special.gamma(1/g) * scipy.special.gamma(3/g)) / (scipy.special.gamma(2/g)**2)

     
    gamma_matrix = gamma_calculation(dct_diff5x5, r, mblock, g)
    

    gamma_matrix = gamma_matrix.reshape(mblock, mblock)

    freq_bands = np.zeros((1, mblock**2))
    freq_bands = zigzag(gamma_matrix) 

    lf_gamma5x5 = freq_bands[1:np.int((mblock**2-1)/3)+1]
    mf_gamma5x5 = freq_bands[np.int((mblock**2-1)/3)+1:2*np.int((mblock**2-1)/3)+1]
    hf_gamma5x5 = freq_bands[np.int(2*(mblock**2-1)/3)+1:]

    geomean_lf_gam = scipy.stats.mstats.gmean(lf_gamma5x5.T)
    geomean_mf_gam = scipy.stats.mstats.gmean(mf_gamma5x5.T)
    geomean_hf_gam = scipy.stats.mstats.gmean(hf_gamma5x5.T)

    mean_dc = np.mean(dct_diff5x5[:, 0])
    
    return mean_dc, geomean_lf_gam, geomean_mf_gam, geomean_hf_gam