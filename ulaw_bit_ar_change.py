import os
import glob


files = glob.glob("/home/ubuntu/Navaneetha/datasets/processed_al_en/alifya-23/wavs_22k/*.wav")
len(files)
8367
def move(path):
     in_path = path
     temp_path = path.replace(".wav","_temp.wav")
     out_path = path.replace("Navaneetha/datasets/processed_al_en/alifya-23/wavs_22k","yash/spoof_data/alifia_dummy_ulaw")
     os.system("sox {} -r 8000 -c 1 -e u-law {}".format(in_path,temp_path))
     os.system("sox {} -r 16000 -b 16 -t wavpcm {}".format(temp_path, out_path))
     os.system("rm ()".format(temp_path))

from multiprocessing import Pool
with Pool(12) as p:
     print(p.map(move , files[0:500]))                     
