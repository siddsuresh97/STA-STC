import os
import pickle
import numpy as np

from zca import ZCA    #"https://github.com/mwv/zca"
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

from collections import Counter

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import skimage

def load_data(spk_fname,feature_fname,time_fname):
    spk = pickle.load(open(spk_fname,"rb"))
    feats = pickle.load(open(feature_fname,"rb"))
    time = pickle.load(open(time_fname,"rb"))
    return spk, feats, time


def bin_neural_video(
        spk_data,
        video_feats,
        time,
        firing_rate=1017.2526245117188,
        vid_length=5,
        display_length=2,
        zero_index=305,
        last_index=2341, lfp = False):
    """Bin neural data according to time."""
    n_frames = len(video_feats)
    n_req_frames = int((n_frames/vid_length)*display_length)                #number of frames in the first 2 seconds
    bin_size = int((1.0/n_req_frames)*firing_rate)
    spk_req_data = spk_data[zero_index:last_index]               # (need these datapoints which correspond to time between 0 and 2 seconds)
    time_req = time[zero_index:last_index]
    video_feats_req = video_feats[:n_req_frames]
    spk_count = []
    if lfp:
        for i in range(1, n_req_frames + 1):
            spk_count += [(np.mean(spk_req_data[int((i-1)*bin_size):int(i*bin_size)]))]
    else:
        for i in range(1, n_req_frames + 1):
            spk_count += [int(sum(spk_req_data[int((i-1)*bin_size):int(i*bin_size)])/np.floor(firing_rate))]
    spk_count = np.asarray(spk_count).reshape(-1, 1)
    return spk_count, video_feats_req


def flatten_feats(feats):
    """Turn feature tensor into matrix."""
    return feats.reshape(feats.shape[0], -1)


def sta(video_feats, spk_data, time, preprocess=False, h = 14, w = 14, lfp = False):
    """STA of video feats and spk_data."""
    if preprocess:
        spk_data, video_feats = bin_neural_video(
            spk_data=spk_data,
            video_feats=video_feats,
            time=time , lfp = lfp)
        video_feats = flatten_feats(video_feats)
    return np.mean((video_feats * spk_data), axis = 0).reshape(h,w)

    '''
    #import ipdb;ipdb.set_trace()
    img = np.mean(np.array(final),axis=0)
    plt.imshow(img.reshape(14,14),cmap="Reds")
    dir = "/home/siddharth/Desktop/test_sta_2"
    savefig("{}/{}.png".format(dir,count))
    plt.close()
    '''


def stc(video_feats, spk_data, time, preprocess=False, h = 14, w = 14, lfp = False):
    feat_shape = video_feats.shape
    if preprocess:
        spk_data, video_feats = bin_neural_video(
            spk_data=spk_data,
            video_feats=video_feats,
            time=time, lfp = lfp)
        video_feats = flatten_feats(video_feats)
    regr = linear_model.LinearRegression()
    regr.fit(video_feats, spk_data)
    #import ipdb;ipdb.set_trace()
    return regr.coef_[:, :].reshape(h,w)  # replace 14 with height/width from activities

def images_after_zca(video_feats):
    for i in video_feats.keys():
        video_feats

def rgb_to_grayscale(video_feats, height , width , channels = 3):
    n = video_feats.shape[0]
    frames_rgb = video_feats.reshape(n, height, width , channels)
    frames_grayscale = []
    for frame in frames_rgb:
        frames_grayscale += [skimage.color.rgb2gray(frame)]
    return np.asarray(frames_grayscale)

def zca_whiten(X):
    '''
    input: an array of shape (n_samples,features)
    output : the array of the same shape as the input but with zca_whitening
    '''
    trf = ZCA().fit(X)
    X_whitened = trf.transform(X)
    X_reconstructed = trf.inverse_transform(X_whitened)
    assert(np.allclose(X, X_reconstructed))
    return X_whitened

def spk_sta_stc(height, width, spk_fname, feats_fname, time_fname="/media/cifs-serrelab/sid/monkey/time.p", lfp = False, run_sta=True, run_stc=True):
    spk, feats, time = load_data(spk_fname, feats_fname, time_fname)
    return backbone(height, width, spk, feats, time, lfp = False, run_sta=True, run_stc=True)

def lfp_sta_stc(height, width, spk_fname, feats_fname, time_fname="/media/cifs-serrelab/sid/monkey/time.p", lfp = True, run_sta=True, run_stc=True):
    spk, feats, time = load_data(spk_fname, feats_fname, time_fname)
    return backbone(height, width, spk, feats, time, lfp = True, run_sta=True, run_stc=True )

def backbone(height, width, spk, feats, time, lfp, run_sta, run_stc):
        time = time['time']
        all_feats, all_spikes, stas, stcs, frame_regressor, video_regressor = [], [], [], [], [], []
        temp_all_feats = []
        temp_all_spikes = []    #to list out the data incase the video was shown multiple times in a session
        for key in spk.keys():
            for data in spk[key]:
                temp_all_spikes += [data]
                temp_all_feats += [feats[key]]
        assert(len(temp_all_feats) == len(temp_all_spikes))
        for idx, i in enumerate(range(len(temp_all_feats))):
            spk_data, video_feats = bin_neural_video(
                spk_data = temp_all_spikes[i],
                video_feats = temp_all_feats[i],
                time=time, lfp = lfp)
            gray_scale_frames = rgb_to_grayscale(video_feats, height = height, width = width, channels = 3)
            flat_feats = flatten_feats(gray_scale_frames)
            #zca_flat_feats = zca_whiten(flat_feats)
            all_feats += [flat_feats]
            all_spikes += [spk_data]
            frame_regressor += [np.arange(len(spk_data))]
            video_regressor += [np.repeat(idx, len(spk_data))]
            if run_sta:
                stas += [sta(video_feats=flat_feats, spk_data=spk_data, time=time, h = height, w = width, lfp = lfp)]
            if run_stc:
                stcs += [stc(video_feats=flat_feats, spk_data=spk_data, time=time, h = height, w = width, lfp = lfp)]
        cat_feats = np.concatenate(all_feats, axis=0)
        cat_spikes = np.concatenate(all_spikes, axis=0)
        cat_frames = np.concatenate(frame_regressor, axis=0)
        cat_videos = np.concatenate(video_regressor, axis=0)

        return stas, stcs, cat_feats, cat_spikes, cat_frames, cat_videos

def lfp_or_spk(height, width, spk_fname, feats_fname, time_fname, lfp, run_sta=True, run_stc=True):

    if(lfp == True):
        stas, stcs, cat_feats, cat_spikes, cat_frames, cat_videos = lfp_sta_stc(height, width, spk_fname,
                                                                                feats_fname,
                                                                                time_fname,
                                                                                lfp , run_sta, run_stc)
    elif(lfp == False):
        stas, stcs, cat_feats, cat_spikes, cat_frames, cat_videos = spk_sta_stc(height, width, spk_fname,
                                                                                feats_fname,
                                                                                time_fname,
                                                                                lfp , run_sta, run_stc)

    else:
        print("Error, please choose a mode(lfp = true or false")

    return stas, stcs, cat_feats, cat_spikes, cat_frames, cat_videos

def stc_all_videos(file, results_dir, gs, main_dict, height, width ):
    for idx,i in enumerate(main_dict.keys()):
        cat_feats = main_dict[i]['cat_feats']
        spk_cat_spikes = main_dict[i]['spk_cat_spikes']
        lfp_cat_lfps = main_dict[i]['lfp_cat_lfps']
        zca_feats = zca_whiten(cat_feats)
        regr1 = linear_model.LinearRegression()
        regr1.fit(zca_feats, spk_cat_spikes)
        total_stc = regr1.coef_
        plt.subplot(gs[int(idx/3),int(idx%3)])                                    #3 corresponds to the number of columns
        plt.title("STC neuron corresponding to device {}".format(i),fontsize=1)
        plt.imshow(regr1.coef_.reshape(height, width),cmap="Reds")
    plt.savefig(os.path.join(results_dir,"stc_all_videos_{}.png".format(file)))

def sta_per_video_avg(file, results_dir, gs, main_dict, height, width):
    for idx,i in enumerate(main_dict.keys()):
        cat_feats = main_dict[i]['cat_feats']
        spk_cat_spikes = main_dict[i]['spk_cat_spikes']
        lfp_cat_lfps = main_dict[i]['lfp_cat_lfps']
        cat_frames = main_dict[i]['cat_frames']
        cat_videos = main_dict[i]['cat_videos']
        zca_feats = zca_whiten(cat_feats)
        total_sta = zca_feats*spk_cat_spikes
        per_video_sta = np.array([np.mean(total_sta[np.argwhere(cat_videos==x),:], axis = 0) for x in np.unique(cat_videos)])
        plt.subplot(gs[int(idx/3),int(idx%3)])
        plt.title("PER_vid_STA_avg corresponding to device {}".format(i),fontsize=5)
        plt.imshow(np.mean(per_video_sta, axis = 0 ).reshape(height, width),cmap="Reds")
    plt.savefig(os.path.join(results_dir,"sta_per_video_avg{}.png".format(file)))

def stc_per_video_avg(file, results_dir, gs, main_dict, height, width):
    for idx,i in enumerate(main_dict.keys()):
        cat_feats = main_dict[i]['cat_feats']
        spk_cat_spikes = main_dict[i]['spk_cat_spikes']
        lfp_cat_lfps = main_dict[i]['lfp_cat_lfps']
        cat_frames = main_dict[i]['cat_frames']
        cat_videos = main_dict[i]['cat_videos']
        zca_feats = zca_whiten(cat_feats)
        total_sta = zca_feats*spk_cat_spikes
        per_video_stc =[np.array(zca_feats[np.argwhere(cat_videos==x),:]) for x in np.unique(cat_videos)]
        feats_per_video = np.array([per_video_stc[i].squeeze() for i in range(len(per_video_stc))])
        per_video_spk = [np.array(spk_cat_spikes[np.argwhere(cat_videos==x),:]) for x in np.unique(cat_videos)]
        spikes_per_video = np.array([per_video_spk[i].reshape(per_video_spk[i].shape[0],1) for i in range(len(per_video_spk))])
        assert(len(per_video_spk)==len(feats_per_video))
        stc_per_video = []
        for i in range(len(feats_per_video)):
            regr = linear_model.LinearRegression()
            regr.fit(feats_per_video[i], spikes_per_video[i])
            stc_per_video += [regr.coef_[:,:]]
        stc_mean = np.mean(np.array(stc_per_video).squeeze() ,axis = 0)
        plt.subplot(gs[int(idx/3),int(idx%3)])
        plt.title("PER_vid_STC_avg corresponding to device {}".format(i),fontsize=5)
        plt.imshow(stc_mean.reshape(height, width),cmap="Reds")
    plt.savefig(os.path.join(results_dir,"stc_per_video_avg{}.png".format(file)))

def per_session_device_data_to_pickle(date, height, width):
    h = height
    w = width
    spike_files_dir = "/media/data_cifs/sid/monkey/spike_lfp_data/spike/{}".format(date)
    lfp_files_dir = "/media/data_cifs/sid/monkey/spike_lfp_data/lfp/{}".format(date)
    device_fnames = os.listdir(spike_files_dir)
    present_devices = [i.split('_')[3].split(".")[0] for i in device_fnames]
    unique_devices = [int(i) for i in present_devices if Counter(present_devices)[i]<2]
    lfp_dict = {}
    spk_dict = {}
    main_dict = {}
    for i in tqdm(unique_devices):
        spk_fname = os.path.join(spike_files_dir, "fname_to_spk_{}.p".format(i))
        lfp_fname = os.path.join(lfp_files_dir, "fname_to_lfp_{}.p".format(i))
        feats_fname = "/media/data_cifs/sid/monkey/frames_to_numpy_corrected.p"
        time_fname = "/media/data_cifs/sid/monkey/time.p"
        lfp_stas, lfp_stcs, lfp_cat_feats, lfp_cat_lfps, lfp_cat_frames, lfp_cat_videos = lfp_or_spk(h, w, lfp_fname, feats_fname, time_fname, lfp = True, run_sta = True, run_stc = True)
        spk_stas, spk_stcs, spk_cat_feats, spk_cat_spikes, spk_cat_frames, spk_cat_videos = lfp_or_spk(h, w, spk_fname, feats_fname, time_fname, lfp = False, run_sta = True, run_stc = True)
        assert(np.allclose(lfp_cat_feats, spk_cat_feats ))
        assert(np.allclose(lfp_cat_frames, spk_cat_frames))
        assert(np.allclose(lfp_cat_videos, spk_cat_videos))
        main_dict.update({i:{'lfp_stas' : lfp_stas, 'lfp_stcs' : lfp_stcs, 'cat_feats' : lfp_cat_feats,
                                        'spk_stas' : spk_stas, 'spk_stcs' : spk_stcs, 'lfp_cat_lfps': lfp_cat_lfps,
                                        'spk_cat_spikes' : spk_cat_spikes, 'cat_frames' : spk_cat_frames, 'cat_videos' : spk_cat_videos}})
    pickle.dump(main_dict,open("/media/data_cifs/sid/monkey/spike_lfp_data/session_data/{}.p".format(date),'wb'))

def visualisation_of_results(file, results_dir, main_dict, height, width):
    h = height
    w = width
    gs = gridspec.GridSpec(7,3)
    stc_per_video_avg(file, results_dir, gs, main_dict, h, w)
    stc_all_videos(file, results_dir, gs, main_dict, h, w)
    sta_per_video_avg(file, results_dir, gs, main_dict, h, w)

def save_per_session_results(dir, results_dir, height, width):
    fnames = os.listdir(dir)
    for file in fnames:
        main_dict = pickle.load(open(os.path.join(dir,file),'rb'))
        visualisation_of_results(file, results_dir, main_dict, height, width)

def main(run_sta=True, run_stc=True, main_dict_exists = True):
    h = 14
    w = 25
    PER_SESSION_DATA_EXISTS = True
    PER_SESSION_RESULTS_EXISTS = True
    '''
    go though spikes of all neurons followed by lfps of all neurons
    '''
    '''
    if(main_dict_exists == False):
        spike_files_dir = "/media/data_cifs/sid/monkey/spike_lfp_data/spike"
        lfp_files_dir = "/media/data_cifs/sid/monkey/spike_lfp_data/lfp"
        present_devices = [6, 7, 9, 11, 13, 20, 22, 24, 25, 26, 28, 29, 31, 32]
        lfp_dict = {}
        spk_dict = {}
        main_dict = {}
        for i in tqdm(present_devices):
            spk_fname = os.path.join(spike_files_dir, "fname_to_spk_{}.p".format(i))
            lfp_fname = os.path.join(lfp_files_dir, "fname_to_lfp_{}.p".format(i))
            feats_fname = "/media/data_cifs/sid/monkey/frames_to_numpy_corrected.p"
            time_fname = "/media/data_cifs/sid/monkey/time.p"
            lfp_stas, lfp_stcs, lfp_cat_feats, lfp_cat_lfps, lfp_cat_frames, lfp_cat_videos = lfp_or_spk(h, w, lfp_fname, feats_fname, time_fname, lfp = True, run_sta = True, run_stc = True)
            spk_stas, spk_stcs, spk_cat_feats, spk_cat_spikes, spk_cat_frames, spk_cat_videos = lfp_or_spk(h, w, spk_fname, feats_fname, time_fname, lfp = False, run_sta = True, run_stc = True)
            assert(np.allclose(lfp_cat_feats, spk_cat_feats ))
            assert(np.allclose(lfp_cat_frames, spk_cat_frames))
            assert(np.allclose(lfp_cat_videos, spk_cat_videos))
            main_dict.update({i:{'lfp_stas' : lfp_stas, 'lfp_stcs' : lfp_stcs, 'cat_feats' : lfp_cat_feats,
                                    'spk_stas' : spk_stas, 'spk_stcs' : spk_stcs, 'lfp_cat_lfps': lfp_cat_lfps,
                                    'spk_cat_spikes' : spk_cat_spikes, 'cat_frames' : spk_cat_frames, 'cat_videos' : spk_cat_videos}})
    else:
        main_dict = pickle.load(open("/media/data_cifs/sid/monkey/spike_lfp_data/individual_device_spk_lfp.p","rb"))
    '''
    per_session_dir = "/media/data_cifs/sid/monkey/spike_lfp_data/session_data"
    results_dir = "/media/data_cifs/sid/monkey/spike_lfp_data/session_data_results"
    if(PER_SESSION_DATA_EXISTS):
        pass
    else:
        dates = ['180716','180719','180724','180728','180730']
        for date in dates:
            per_session_device_data_to_pickle(date, h, w)
    if(PER_SESSION_RESULTS_EXISTS):
        pass
    else:
        save_per_session_results(per_session_dir, results_dir, h, w)






if __name__=="__main__":
    main()
