import mne
from tqdm import tqdm
import numpy as np

import random
from scipy import signal
#from meegkit.detrend import detrend

import pandas as pd
from sklearn import preprocessing



def get_label_normal(file):
    label = file.split("/")[-3]
    if label == "normal":
        return [0]
    elif label == "abnormal":
        return [1]

def get_label_epilepsy(file):
    label = file.split("/")[-5]
    if label == "no_epilepsy_edf":
        return [0]
    elif label == "epilepsy_edf":
        return [1]
    
def get_label_seizure(file):
    csv_meta = pd.read_csv("../seizure_raw/seizures.csv")

    seizures = ["absz", "cpsz", "fnsz", "gnsz", "mysz", "spsz", "tcsz", "tnsz"]
    class_csv = csv_meta[csv_meta["id"] == file.split("/")[-1].split(".")[0]]["class_code"]
    label = None
    for s in seizures:
        if s in class_csv.to_string():
            label = s
    
    if label == "absz":
        return [0]
    elif label == "cpsz":
        return [1]
    elif label == "fnsz":
        return [2]
    elif label == "gnsz":
        return [3]
    elif label == "mysz":
        return [4]
    elif label == "spsz":
        return [5]
    elif label == "tcsz":
        return [6]
    elif label == "tnsz":
        return [7]
    else:
        return [8]

def get_label_dementia(file):
    files_data = pd.read_csv('../../../../../ds004504/participants.tsv', sep="\t")
    idx = file.split("/")[-3]
    label = np.array(files_data[files_data["participant_id"] == idx]["Group"])[0]
    if label == "C":
        return [0]
    elif label == "A":
        return [1]
    elif label == "F":
        return [1]

def multichannel_sliding_window(X, size, step):
    shape = (X.shape[0] - X.shape[0] + 1, (X.shape[1] - size + 1) // step, X.shape[0], size)
    strides = (X.strides[0], X.strides[1] * step, X.strides[0], X.strides[1])
    return np.lib.stride_tricks.as_strided(X, shape, strides)[0]


def normalize(data: np.ndarray, dim=1, norm="l2") -> np.ndarray:
    """Normalizes the data channel by channel

    Args:
        data (np.ndarray): Raw EEG signal
        dim (int, optional): 0 columns, 1 rows. The dimension where 
        the mean and the std would be computed.

    Returns:
        np.ndarray: Channel-wise normalized matrix
    """
    normalized_data = preprocessing.normalize(data, axis=dim, norm=norm)
    return normalized_data

    
def read_edf_file(file, use_windows=True, window_size=100, num_windows=100): 
    """
    output array of shape #num windows, #num of channels, #window size
    """
    try:
        data_raw = mne.io.read_raw_edf(file, preload=True)
        channels_to_use = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
                                  "T3", "C3", "CZ", "C4", "T4", "T5",
                                  "P3", "PZ", "P4", "T6", "O1", "O2"]
        ch_name_update_func = lambda ch: ch.split(' ')[-1].split('-')[0]
        data_raw.rename_channels(mapping=ch_name_update_func)
        data_raw = data_raw.pick_channels(channels_to_use, ordered=True)
        montage = mne.channels.make_standard_montage('standard_1020')
        data_raw.set_montage(montage, match_case=False, match_alias=True)
        data_raw.filter(l_freq=1.0, h_freq=45.0)
        data_raw.resample(100, npad='auto')

        data_raw = data_raw.get_data()*1e6
          
        data_raw = multichannel_sliding_window(data_raw, window_size, window_size)
        
        data_raw = np.clip(data_raw, a_min=-800, a_max=800)
        
        data_epochs_normalized = []
        for epoch in data_raw:
            epoch = normalize(epoch)
            data_epochs_normalized.append(epoch)
        data_raw = np.array(data_epochs_normalized)
        
    except:
        return 

    startpoint = 1
    if data_raw.shape[0] >= num_windows+startpoint:
        return [data_raw[startpoint:num_windows+startpoint, :, :]]
    else:
        return

    
def read_dem_edf(file, use_windows=True, num_windows=100):
    data_raw = mne.io.read_raw(file, verbose=False, preload=True)
    channels_to_use = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
                                  "T3", "C3", "CZ", "C4", "T4", "T5",
                                  "P3", "PZ", "P4", "T6", "O1", "O2"]
    ch_name_update_func = lambda ch: ch.split(' ')[-1].split('-')[0]
    data_raw.rename_channels(mapping=ch_name_update_func)
    data_raw = data_raw.pick_channels(channels_to_use, ordered=True)
    montage = mne.channels.make_standard_montage('standard_1020')
    data_raw.set_montage(montage, match_case=False, match_alias=True)
    
    data_raw = mne.make_fixed_length_epochs(data_raw, duration=1.0, overlap=0.5)
    data_raw.filter(l_freq=1.0, h_freq=45.0)
    data_raw.resample(100, npad='auto')
    data_raw = data_raw.get_data()*1e6
    
    raws = []
    print(data_raw.shape)
    for i in range(0, 100, 20):
        if data_raw.shape[0] > (i+num_windows):
            raws.append(data_raw[i:(i+num_windows)])
    if not use_windows:
        raws = [raws[int(len(raws)/2)]]
    return raws
    
def build_data(raw_data, use_windows=True, window_size=100, num_windows=100, dataset="tuh"):
    
    all_data_features = []
    data_labels = []
    
    for file in tqdm(raw_data):
        if dataset == "tuh" or dataset == "epilepsy" or dataset == "nmt":
            edf_data = read_edf_file(file, use_windows=use_windows, window_size=window_size, num_windows=num_windows)
        elif dataset == "dementia":
            edf_data = read_dem_edf(file, use_windows=use_windows, window_size=window_size, num_windows=num_windows)
        elif dataset == "seizure":
            try:
                edf_data = pd.read_parquet(file, engine='pyarrow')
                edf_data = np.array(edf_data)[:, :-1].T
                edf_data = multichannel_sliding_window(edf_data, window_size, window_size)
            except:
                continue
        
        if edf_data is None:
            continue
        else:
            edf_data = np.array(edf_data).squeeze()
        #else:
        #    for edf_data1 in edf_data:
        all_data_features.append(edf_data)
        
        if dataset == "tuh" or dataset == "nmt":
            label = get_label_normal(file)
        elif dataset == "epilepsy":
            label = get_label_epilepsy(file)
        elif dataset == "dementia":
            label = get_label_dementia(file)
        elif dataset == "seizure":
            label = get_label_seizure(file)
            if label is None:
                continue
        data_labels.append(np.array(label))
    
    all_data_features = np.array(all_data_features)
    data_labels = np.array(data_labels)
            
    return all_data_features, data_labels