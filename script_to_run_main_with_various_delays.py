from subprocess import call
import os


def main():
    result_dir = "/media/data_cifs/sid/monkey/spike_lfp_all/lfp_results"
    for delay in range(0,110,10):
        path = os.path.join(result_dir,"delay_{}".format(delay))
        if(os.path.isdir(path)):
            pass
        else:
            os.makedirs(path)
        call(["python3","main.py","--delay","{}".format(delay),"--result_dir", path])


if __name__=="__main__":
    main()
