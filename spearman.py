
from spike_data_familiar_videos import load_data_from_h5
import numpy as np
import scipy.stats
def bin_neural_video_modified_for_spearman(spk_data,
                     firing_rate=1017.2526245117188,
                     vid_length=5,
                     display_length=2,
                     zero_index=305,
                     last_index=2341, lfp = False):
   """Bin neural data according to time."""
   n_frames = 126
   n_req_frames = int((n_frames/vid_length)*display_length)
    #number of frames in the first 2 seconds
   bin_size = int((1.0/n_req_frames)*firing_rate)
   spk_req_data = spk_data[zero_index:last_index]               # (need these datapoints which correspond to time between 0 and 2 seconds
   spk_count = []
   if lfp:
       for i in range(1, n_req_frames + 1):
           spk_count += [(np.mean(spk_req_data[int((i-1)*bin_size):int(i*bin_size)]))]
   else:
       for i in range(1, n_req_frames + 1):
           spk_count += [int(sum(spk_req_data[int((i-1)*bin_size):int(i*bin_size)])/np.floor(firing_rate))]
       spk_count = np.asarray(spk_count).reshape(-1, 1)
   return spk_count

def dict_with_spk_count(main_dict):
    temp={}
    for movie in main_dict.keys():
        a=[]
        for i in main_dict[movie]:
            a.append(bin_neural_video_modified_for_spearman(i))
        temp.update({movie:np.array(a)})
    return temp



def dict_for_spearman(device_id, data_spk):
    main_dict={}
    for date in data_spk.keys():
        familiar_trials_spk = [i for i in range(len(data_spk[date]['trial_info'])) if data_spk[date]['trial_info']['stim_familiarized'][i]==1 ]
        familiar_fnames_spk = [data_spk[date]['trial_info']['stim_names'][i] for i in familiar_trials_spk]
        spk_data_familiar_trials = data_spk[date]['data'][familiar_trials_spk,:,device_id]
        for i in range(len(spk_data_familiar_trials)):
            if(familiar_fnames_spk[i] in main_dict.keys()):
                main_dict[familiar_fnames_spk[i]].append(spk_data_familiar_trials[i])
            else:
                main_dict.update({familiar_fnames_spk[i]:[spk_data_familiar_trials[i]]})
    return main_dict

def spearman_coef(main_dict):
    import ipdb;ipdb.set_trace()
    score = []
    for key in main_dict.keys():
        Z = main_dict[key]
        score += [np.mean(np.tril(scipy.stats.spearmanr(Z.squeeze().reshape(50,Z.shape[0])), k=-1))]
    score =  [i for i in score if np.isnan(i)==False]
    return np.mean(np.array(score))

def main():
    dir_data = '/media/data_cifs/sid/monkey/spike_data/data/'
    animal_name = 'thor'
    task = 'movies'
    device_id = 6
    dates = []
    data_spk = load_data_from_h5(dir_data, animal_name, task, 'spk', dates)
    main_dict = dict_for_spearman(device_id, data_spk)
    req_dict = dict_with_spk_count(main_dict)
    coef = spearman_coef(req_dict)
    print(coef)

if __name__ == "__main__":
    main()
