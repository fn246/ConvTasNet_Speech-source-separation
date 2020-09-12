#!/usr/bin/env python
# Created on 2018/12
# Author: Kaituo XU

#import argparse
import json
import os

import librosa


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=16000):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)




#fatemeh#def preprocess(args):
  #fatemeh#  for data_type in ['tr', 'cv', 'tt']:
  #fatemeh#      for speaker in ['mix', 's1', 's2']:
  #fatemeh#          preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),
  #fatemeh#                             os.path.join(args.out_dir, data_type),
  #fatemeh#                             speaker,
  #fatemeh#                             sample_rate=args.sample_rate)
            
#fatemeh delete args
def preprocess(in_dir,out_dir,sample_rate):
    for data_type in ['tr', 'cv', 'tt']:
        for speaker in ['mix', 's1', 's2']:
            preprocess_one_dir(os.path.join(in_dir, data_type, speaker),
                               os.path.join(out_dir, data_type),
                               speaker,
                               sample_rate=sample_rate)
#%%
if __name__ == "__main__":
    in_dir="D:/Amir Kabir University/thesis/conv-tasnet/Conv-TasNet-master/data"

    out_dir="D:/Amir Kabir University/thesis/conv-tasnet/Conv-TasNet-master/outdata"
    sample_rate=16000
    preprocess(in_dir,out_dir,sample_rate)
    #fatemeh#parser = argparse.ArgumentParser("WSJ0 data preprocessing")
    #fatemeh#parser.add_argument('--in-dir', type=str, default=None,
    #fatemeh#                    help='Directory path of wsj0 including tr, cv and tt')
    #fatemeh#parser.add_argument('--out-dir', type=str, default=None,
    #fatemeh#                    help='Directory path to put output files')
    #fatemeh#parser.add_argument('--sample-rate', type=int, default=8000,
    #fatemeh#                    help='Sample rate of audio file')
    #fatemeh#args = parser.parse_args()
    #fatemeh#print(args)
    #fatemeh#preprocess(args)
