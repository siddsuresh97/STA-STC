# STA-STC
STA,STC on neural spikes

Results on :
https://drive.google.com/folderview?id=1Uni0YQN4F5uBFb8PednIRWkr7GCSLZJ1


MAKE A BACKUP OF YOUR DATA INCASE SOMETHING GOES WRONG

1) Use vid_to_frames in the repository 'intermediate-layer-activations' to break teh video down into frames.
switch the h and w arguments . For example if you want frames of dimension (14,25,3) ,run the script 

python vid_to_frames.py --dir PATH/TO/DIR --ext '.mp4' --h 25 --w 14 --fps 50

Here PATH/TO/DIR is the path to the directory containing the videos


2) python3 frames_to_numpy.py

Stores the frames obtained after running the previous script.
dir is the directory which was specified in PATH/TO/DIR
ext_last_char the last character of the extension of the image. 'g' if the images are of jpeg format
output_dir is the directory where you want to save the pickle file containing the a dictionary with video filenames as keys and the frames(covnerted to numpy) as values.
pickle_fname is the name of the output

3) python3 spike_data_familiar
Set the following flags and variables.
data_exists
dir_data
animal_name
task
familiar
unfamiliar
spk_data_fname
lf_data_fname


4) python3 main.py
feats_fname -> path to the pickle file obtained in 2
store_result_in ->
spike_files_dir
lfp_files_dir
