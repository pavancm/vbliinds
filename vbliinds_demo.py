from vbliinds_frame_numba import temporal_dc_variation_feature_extraction_frame,\
NSS_spectral_ratios_feature_extraction_frame,motion_feature_extraction_frame,\
compute_niqe_features
import scipy.stats
import skvideo.io
import cv2
import argparse
import numpy as np

def main(args):
    filepath = args.vid_path
    vid = skvideo.io.FFmpegReader(filepath)
    T, height, width, C = vid.getShape()
    
    #Declare arrays
    std_dc,mean_dc = np.zeros(T-1),np.zeros(T-1)
    geomean_lf_gam, geomean_mf_gam, geomean_hf_gam = np.zeros(T-1),np.zeros(T-1),\
    np.zeros(T-1)
    motion_diff, mode10x10 = np.zeros(T-1), np.zeros(T-1)
    Coh10x10 = []
    niqe_feat = np.zeros((T,37),dtype=np.float32)
    
    videoData = np.zeros((2, height, width), dtype=np.uint8)
    prev_frame = next(vid)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    niqe_feat[0,:] = compute_niqe_features(prev_frame)
    
    for i in range(T-1):
        curr_frame = next(vid)
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        videoData[0,:,:] = prev_frame
        videoData[1,:,:] = curr_frame
        
        niqe_feat[i+1,:] = compute_niqe_features(curr_frame)
        std_dc[i] = temporal_dc_variation_feature_extraction_frame(videoData)
        mean_dc[i], geomean_lf_gam[i], geomean_mf_gam[i], geomean_hf_gam[i] = \
        NSS_spectral_ratios_feature_extraction_frame(videoData)
        C, motion_diff[i], mode10x10[i] = motion_feature_extraction_frame(videoData)
        Coh10x10.append(C)
        
        prev_frame = curr_frame
    
    dt_dc_temp = np.abs(std_dc[1:] - std_dc[:-1])
    dt_dc_measure1 = np.mean(dt_dc_temp)
    
    geo_high_ratio = scipy.stats.mstats.gmean(geomean_hf_gam/(0.1 + (geomean_mf_gam + geomean_lf_gam)/2))
    geo_low_ratio = scipy.stats.mstats.gmean(geomean_mf_gam/(0.1 + geomean_lf_gam))
    geo_HL_ratio = scipy.stats.mstats.gmean(geomean_hf_gam/(0.1 + geomean_lf_gam))
    geo_HM_ratio = scipy.stats.mstats.gmean(geomean_hf_gam/(0.1 + geomean_mf_gam))
    geo_hh_ratio = scipy.stats.mstats.gmean(((geomean_hf_gam + geomean_mf_gam)/2)/(0.1 + geomean_lf_gam))
    
    dt_dc_measure2 = np.mean(np.abs(mean_dc[1:] - mean_dc[:-1]))
    
    Coh10x10 = np.array(Coh10x10)
    meanCoh10x10 = np.mean(Coh10x10)
    G = np.mean(motion_diff) / (1 + np.mean(mode10x10))
    
    bliinds_feat = np.hstack((np.mean(niqe_feat,axis=0),\
                              np.log(1+dt_dc_measure1),np.log(1+dt_dc_measure2),\
                              np.log(1+geo_HL_ratio), np.log(1+geo_HM_ratio), \
                              np.log(1+geo_hh_ratio), np.log(1+geo_high_ratio), \
                              np.log(1+geo_low_ratio),np.log(1+meanCoh10x10),\
                              np.log(1+G)))
    print(bliinds_feat)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--vid_path', type=str, default='test.mp4', \
                        help='Path to video', metavar='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
