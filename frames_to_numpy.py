
'''
This script stores all the features from the frames of all videos in a pickle file
'''
import os
import cv2
import pickle
import numpy as np
'''
Data format
dir:
    -1
        -franem1
        -frmae2
        -...
    -2
        -frame1
        -frame2
        -...
directory structure after running the vid_to_frames script
'''
def load_dict(dir="", ext_last_char = "g", h = 14, w = 25):
    result = {}
    dirnames = os.listdir(dir)
    for idx,i in enumerate(dirnames):
        target_dir = os.path.join(dir,i)
        frame_fnames = [j for j in os.listdir(target_dir) if j[-1]==ext_last_char]           #the frames are of jpeg format and there is the original movie file in the target_dir (dir structure after running vid_toFrames.py)
        frame_fnames.sort()
        frames=[]
        for frame in frame_fnames:
            frame_path = os.path.join(target_dir,frame)
            frames += [cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)]
        frames = np.asarray(frames).reshape(-1, h, w, 3)
        result.update({i:frames})
        print("{} images done".format(idx))
    return result

def save_to_pickle(output_dir = "/media/data_cifs/sid/monkey", dict = dict, pickle_fname="frames_to_numpy_corrected.p"):
    fname = os.path.join(output_dir,pickle_fname)
    pickle.dump(dict, open(fname, "wb"))

def main():
    #dir = "/media/data_cifs/sid/monkey/450_vids"
    dict = load_dict(dir = "/media/data_cifs/sid/monkey/3_sec_frames", ext_last_char = 'g', h = 14, w = 25)
    save_to_pickle(output_dir="/media/data_cifs/sid/monkey/3_sec_m4v", dict = dict, pickle_fname = "frames_to_numpy_unfamiliarised.p")
    print("FINALLY DONE")


if __name__=="__main__":
    main()
