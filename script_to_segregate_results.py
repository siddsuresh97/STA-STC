import os
import shutil
result_dir = "/media/data_cifs/sid/monkey/spike_lfp_all/lfp_results"

delay_dir = os.listdir(result_dir)
dates = [180716,180719,180724,180728,180730]

for dir in delay_dir:
	main_dir = os.path.join(result_dir,dir)
	for date in dates:
		path = os.path.join(main_dir,str(date))
		if(os.path.isdir(path)):
			pass
		else:
			os.makedirs(path)
	for item in os.listdir(main_dir):
		if(os.path.isdir(os.path.join(main_dir,item))):
			continue
		if(item[-7]=='6'):
			shutil.move(os.path.join(main_dir,item),os.path.join(main_dir,"180716"))
		if(item[-7]=='9'):
			shutil.move(os.path.join(main_dir,item),os.path.join(main_dir,'180719'))
		if(item[-7]=='4'):
			shutil.move(os.path.join(main_dir,item),os.path.join(main_dir,'180724'))
		if(item[-7]=='8'):
			shutil.move(os.path.join(main_dir,item),os.path.join(main_dir,'180728'))
		if(item[-7]=='0'):
			shutil.move(os.path.join(main_dir,item),os.path.join(main_dir,"180730"))
