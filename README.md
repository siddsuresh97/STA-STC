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
output_dir is the directory where you want to save the pickle file containing the a dictionary with video  filenames as keys and the frames(covnerted to numpy) as values.    
pickle_fname is the name of the output   

3) python3 spike_data_familiar  
Set the following flags and variables.      
data_exists
dir_data -> directory where the hdf5(from Ruobing) file is stored  
animal_name  
task  
familiar -> set to True if analysis is being performed on only the familiar set of videos  
unfamiliar -> set to True if analysis is being performed on only the unfamiliar set of videos  
boht -> set to True if analysis is being performed on all the videos  
spk_data_fname -> directory where you want to store the spikes data  
lfp_data_fname -> directory where you ant to store the lfp data  



4) python3 main.py  
per_session_dir - > directory where the per session data is stored(useful when per_session_data exists=   True and per_session_results = False)  
feats_fname -> path to the pickle file obtained in step 2  
store_result_in -> the directory where you want to store your per_session data in  
spike_files_dir -> directory where the spike data is stored (same as spk_data_fname from step 3)  
lfp_files_dir ->  directory where the lfp data is stored (same as lfp_data_fname from step 3)  
time_fname = pickle file where your time axis is stored      

5) python script_to_run_main_with_various_delay.py  

This can me used to run the main.py with multiple delays. Set the variables in main.py before running this script.  


6) python script_to_segregate_results

This is used to put the results into appropriate directories. Run this script after the results have been generated from step 5.
