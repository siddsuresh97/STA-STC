import os
import store_hdf5
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter
# empty if load all; otherwise a list of dates, e.g. ['180724','180731']

def load_data_from_h5(dir_data='/shared/homes/rxia/data', name='', block_type='', signal_type='', dates=[]):
    hdf_file_path = '{}/all_data_{}_{}.hdf5'.format(dir_data, name, block_type)
    dict_data = dict()
    if len(dates) == 0:
        dates = list(store_hdf5.ShowH5(hdf_file_path).keys())
    for date in dates:
        print('loading {} {} {}'.format(date,block_type,signal_type))
        dict_data[date] = store_hdf5.LoadFromH5(hdf_file_path, h5_groups=[date, block_type, signal_type])
    return(dict_data)


def familiar_spk_lfp(date, device_id, data_spk, data_lfp):
    familiar_trials_spk = [i for i in range(len(data_spk[date]['trial_info'])) if data_spk[date]['trial_info']['stim_familiarized'][i]==1 ]
    familiar_fnames_spk = [data_spk[date]['trial_info']['stim_names'][i] for i in familiar_trials_spk]
    #both the variables below are similar to the ones above . It means that dta is ordered in a similar way
    familiar_trials_lfp = [i for i in range(len(data_lfp[date]['trial_info'])) if data_lfp[date]['trial_info']['stim_familiarized'][i]==1 ]
    familiar_fnames_lfp = [data_lfp[date]['trial_info']['stim_names'][i] for i in familiar_trials_lfp]
    #req_unit_index_spk = [i for i in data_lfp[date]['signal_info']['channel_index']].index(device_id)
    spk_data_familiar_trials = data_spk[date]['data'][familiar_trials_spk,:,device_id]
    lfp_data_familiar_trials = data_lfp[date]['data'][familiar_trials_lfp,:,device_id]

    fname_to_spk={}
    fname_to_lfp={}

    for i in range(len(familiar_fnames_spk)):
        video_fname = familiar_fnames_spk[i]
        assert(len(spk_data_familiar_trials)==len(familiar_trials_spk))
        if(video_fname in fname_to_spk.keys()):
            fname_to_spk[video_fname].append(spk_data_familiar_trials[i])
        else:
            fname_to_spk.update({video_fname:[spk_data_familiar_trials]})

        assert(len(familiar_fnames_lfp)==len(familiar_fnames_spk))

        if(video_fname in fname_to_lfp.keys()):
            fname_to_lfp[video_fname].append(lfp_data_familiar_trials[i])
        else:
            fname_to_lfp.update({video_fname:[lfp_data_familiar_trials]})

    return fname_to_spk, fname_to_lfp

def data_to_pickle(date, device, data_spk, data_lfp):
    spk_data_fname = "/media/data_cifs/sid/monkey/spike_lfp_data/spike/{}/fname_to_spk_{}.p".format(date, device)
    lfp_data_fname = "/media/data_cifs/sid/monkey/spike_lfp_data/lfp/{}/fname_to_lfp_{}.p".format(date, device)
    if(os.path.isdir("/media/data_cifs/sid/monkey/spike_lfp_data/spike/{}".format(date))):
        pass
    else:
        os.makedirs("/media/data_cifs/sid/monkey/spike_lfp_data/spike/{}".format(date))
    if(os.path.isdir("/media/data_cifs/sid/monkey/spike_lfp_data/lfp/{}".format(date))):
        pass
    else:
        os.makedirs("/media/data_cifs/sid/monkey/spike_lfp_data/lfp/{}".format(date))
    '''
    if(os.path.isdir("/media/data_cifs/sid/monkey/spike_lfp_data/spike/{}".format(date))==False):
        os.makedirs("/media/data_cifs/sid/monkey/spike_lfp_data/spike/{}".format(date))

    if(os.path.isdir("/media/data_cifs/sid/monkey/spike_lfp_data/lfp/{}")==False):
        os.makedirs("/media/data_cifs/sid/monkey/spike_lfp_data/lfp/{}".format(date))
    '''
    device_id = list(data_spk[date]['signal_info']['channel_index']).index(device)
    spk, lfp = familiar_spk_lfp(date, device_id, data_spk, data_lfp)
    pickle.dump(spk, open(spk_data_fname,'wb'))
    pickle.dump(lfp,open(lfp_data_fname,'wb'))

def single_date_data_pickle_dump(date, data_spk, data_lfp):
    devices = [data_spk[date]['signal_info']['channel_index'][i] for i in range(len(data_spk[date]['signal_info']['channel_index']))]
    unique_devices = [i for i in Counter(devices).keys() if Counter(devices)[i]<2]

    for device in unique_devices:
        data_to_pickle(date, device, data_spk, data_lfp)


def main(data_exists = True):
    if(data_exists):
        print("Data already exists. Kindle check in /media/data_cifs/sid/monkey/spike_lfp_data/ ")

    else:
        dir_data = '/media/data_cifs/sid/monkey/spike_data/data/'
        animal_name = 'thor'
        task = 'movies'
        #dates = []
        dates = []
        data_spk = load_data_from_h5(dir_data, animal_name, task, 'spk', dates) # Load spike data
        data_lfp = load_data_from_h5(dir_data, animal_name, task, 'lfp', dates) # Load LFP data

        for date in data_spk.keys():
            single_date_data_pickle_dump(date, data_spk, data_lfp)


if __name__ == "__main__":
    main()
