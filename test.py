import os
import shutil
import librosa
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm
from transformers import Wav2Vec2Config, Wav2Vec2Model
import torch.nn as nn
import torchviz
import torch
from datasets import load_metric
import re
import unicodedata


def preprocess_data(dir_path, des_name, copy_file):
    count = 1
    audio = []
    label = []
    # des_path = dir_path + '/final'
    des_path = dir_path + '/' + des_name
    src_path = dir_path + '/ori_data'
    all_dir = os.listdir(src_path)
    all_dir.remove('record10-2')
    all_dir.sort()
    for e in all_dir:
        # if e != 'record1-1':
        #     continue
        data_path = src_path + f'/{e}'
        audio_path = data_path + '/audio'
        label_path = data_path + '/label.txt'
        with open(label_path,'r',encoding='utf-8')as fp:
            data =  fp.readlines()
        data = [unicodedata.normalize('NFKC', e).strip().replace('.mp3', '') for e in data if e.strip()!='']
        # data = [e.strip().replace('.mp3', '') for e in data if e.strip()!='']
        # print(data)
        # exit()
        for i in range (0, len(data)):
            # if count == 3897:
            #     print(e, data[i])
            # if '雞' in data[i]:
            #     print(e, data[i])
            if i%4 == 0:
                try:
                    if copy_file:
                        shutil.copyfile(audio_path + '/' + data[i]+'.mp3', des_path+'/audio/'+str(count).zfill(5)+'.mp3')
                    audio.append(str(count).zfill(5)+'.mp3')
                    # if '快' in label_temp:
                    # print(data[i+1])
                    for j in string.punctuation:
                        label_temp = data[i+1].replace(j,'')
                    # print(label_temp.replace('（','(').split('(')[-1].replace(')','').replace('）','').replace('.',''))
                    label.append(label_temp.replace('（','(').split('(')[-1].replace(')','').replace('）','').replace('.',''))
                    # label.append(data[i+1].replace('（','(').split('(')[-1].replace(')',''))
                    count += 1
                except:
                    print('error',e, data[i], audio_path + '/' + data[i]+'.mp3')
                    continue
    if copy_file:
        write_data(audio, label, des_name)
        split_data(des_path)
    return (audio, label)


def preprocess_data2(dir_path, copy_file, duplicate):
    sent_audio = []
    sent_label = []
    word_audio = []
    word_label = []
    word_count = 1
    sent_count = 1
    ignore_words = r'[,?.!\-;:"“%\'�，。、？／‑；,]'
    punctuations = '''!(){}:'"\,<>.?@#$%^&*_~。?'''
    dir_path = '../../../../../extra_space1/data/audio_data'
    src_path = '../../../../../extra_space1/data/audio_data/ori_data'
    all_dir = os.listdir(src_path)
    all_dir.remove('record10-2')
    if not duplicate:
        temp = set()
        for e in all_dir:
            temp.add(e[:-2])
        all_dir = [e + '-1' if e != 'record5' else e + '-2' for e in temp]
    all_dir.sort()
    # print(all_dir)
    # exit()
    for e in all_dir:
        # if e != 'record4-1':
        #     continue
        data_path = src_path + f'/{e}'
        audio_path = data_path + '/audio'
        label_path = data_path + '/label.txt'
        with open(label_path,'r',encoding='utf-8')as fp:
            data =  fp.readlines()
        data = [unicodedata.normalize('NFKC', e).strip().replace('.mp3', '') for e in data if e.strip()!='']
        # print(data)
        for i in range (0, len(data)):
            flag = False
            if i%4 == 1:
                # print(data[i])
                # if data[i-1] != '0137':
                #     continue
                temp = data[i].split('(')
                if len(temp) >= 3:
                    temp = [''.join(temp[:-1]).replace(')','')] + [temp[-1]]
                # print(temp)
                for char in temp[0]:
                    if char in punctuations:
                        # print(temp)
                        if len(temp[0]) <= 3:
                            continue
                        try:
                            if copy_file:
                                if not duplicate:
                                    shutil.copyfile(audio_path + '/' + data[i-1]+'.mp3', dir_path + '/sentence/audio/'+str(sent_count).zfill(5)+'.mp3')
                                else:
                                    shutil.copyfile(audio_path + '/' + data[i-1]+'.mp3', dir_path + '/sentence_dup/audio/'+str(sent_count).zfill(5)+'.mp3')
                            sent_audio.append(str(sent_count).zfill(5)+'.mp3')
                        except:
                            print('error1',e, data[i], audio_path + '/' + data[i-1]+'.mp3')
                        label_temp= re.sub(ignore_words, '', temp[-1].split(')')[0])
                        # sent_label.append(audio_path + '/' + data[i-1]+'.mp3' +  data[i])
                        # label_temp= re.sub(ignore_words, '', data[i])
                        sent_label.append(label_temp)
                        sent_count += 1
                        flag = True
                        break
                if flag == False:
                    try:
                        if copy_file:
                            if not duplicate:
                                shutil.copyfile(audio_path + '/' + data[i-1]+'.mp3', dir_path + '/words/audio/'+str(word_count).zfill(5)+'.mp3')
                            else:
                                shutil.copyfile(audio_path + '/' + data[i-1]+'.mp3', dir_path + '/words_dup/audio/'+str(word_count).zfill(5)+'.mp3')
                        word_audio.append(str(word_count).zfill(5)+'.mp3')
                    except:
                        print('error2',e, data[i], audio_path + '/' + data[i-1]+'.mp3')
                    word_label.append(temp[-1].split(')')[0])
                    # word_label.append(audio_path + '/' + data[i-1]+'.mp3' + data[i])
                    word_count += 1
    if copy_file:
        if not duplicate:
            write_data(word_audio, word_label, 'words')
            write_data(sent_audio, sent_label, 'sentence')
            split_data('../../../../../extra_space1/data/audio_data/words')
            split_data('../../../../../extra_space1/data/audio_data/sentence')
        else:
            write_data(word_audio, word_label, 'words_dup')
            write_data(sent_audio, sent_label, 'sentence_dup')
            split_data('../../../../../extra_space1/data/audio_data/words_dup')
            split_data('../../../../../extra_space1/data/audio_data/sentence_dup')
    return (sent_audio, sent_label), (word_audio, word_label)


def seconds_to_hours(seconds):
        hours = seconds // 3600
        remaining_seconds = seconds % 3600
        minutes = remaining_seconds // 60
        remaining_seconds = remaining_seconds % 60
    
        return hours, minutes, remaining_seconds


def cal_total_time(path):
    max, min = 0, 10
    total_time, count = 0, 0
    data_list = os.listdir(path)
    for i in tqdm(range(0, len(data_list))):
        wav, sr = librosa.load(path + f'/{data_list[i]}', sr = 16000)
        duration = librosa.get_duration(y=wav, sr=sr)
        if duration > max:
            max = duration
        if duration < min:
            min = duration
        total_time += duration
        count += 1

    hours, minutes, remaining_seconds = seconds_to_hours(total_time)
    print(f"總時長：{hours} hr, {minutes} min, {remaining_seconds} sec")
    print(count, max, min)


def write_data(audio, label, des):
    print(des)
    result = 'path|transcript\n'
    for i in range(0, len(audio)):
        result += des + '/audio/' + audio[i] + '|' + label[i] + '\n'
    with open(des + '/label.txt', 'w', encoding='utf-8')as f:
        f.write(result)
    f.close()


def split_data(path):
    result = 'path|transcript\n'
    with open(path + '/label.txt', 'r', encoding='utf8')as f:
        data = f.readlines()
        data.pop(0)
    f.close()
    train, test = train_test_split(data,train_size = 0.7, random_state=42)
    test, val = train_test_split(test,train_size = 0.66, random_state=42)
    for e in train:
        result += e
    with open(path + '/train.txt', 'w', encoding='utf-8')as f:
        f.write(result)
    result = 'path|transcript\n'
    for e in test:
        result += e
    with open(path + '/test.txt', 'w', encoding='utf-8')as f:
        f.write(result)
    result = 'path|transcript\n'
    for e in val:
        result += e
    with open(path + '/val.txt', 'w', encoding='utf-8')as f:
        f.write(result)


def evaluate_wer(pred, label):
    wer_metric = load_metric("wer")
    # pred_strs = self.processor.batch_decode(pred)
    # label_strs = self.processor.batch_decode(label, group_tokens=False)
    # wer = self.wer_metric.compute(predictions=pred_strs, references=label_strs)
    result = wer_metric.compute(predictions=pred, references=label)
    return result


'''WER測試'''
# pred = ["hai malakpot kami a misalam", "pasowa ko mato’asay tamiyanan", "kafana’ mingodo to makakay"]
# ref = ["hai malakapot kami a misalama", "pasowal ko mato’asay tamiyanan", "kafana’ mingodo to makakaay"]
# print("WER:", evaluate_wer(pred, ref))

'''將資料統一標號，並彙整到一個資料夾'''
# des_name = "data_6771"
# path = '../../../../../extra_space1/data/audio_data'
# audio, label = preprocess_data(path, des_name, True)
# audio, label = preprocess_data(path, des_name, False)
# print(len(audio), len(label))

'''計算音檔的長度跟數量'''
# all_dir = os.listdir('../../../../../extra_space1/data/audio_data/ori_data')
# all_dir.sort()
# for e in all_dir:
#     print(e)
#     path = f'../../../../../extra_space1/data/audio_data/ori_data/{e}/audio'
#     cal_total_time(path)
# des_name = 'data_6774'
# path = f'../../../../../extra_space1/data/audio_data/{des_name}/audio'
# cal_total_time(path)

'''分成句子跟詞'''
# path = '../../../../../extra_space1/data/audio_data'
# sents, words = preprocess_data2(path, copy_file=True, duplicate=False)
# sents, words = preprocess_data2(path, copy_file=True, duplicate=True)
# sents, words = preprocess_data2(path, copy_file=False, duplicate=True)
# print(len(sents[0]), len(sents[1]), len(words[0]), len(words[1]))


'''測試模型修改'''
# config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
# print(config)
# with open('temp.txt', 'w')as f:
#     f.write(str(config))
# f.close()
# model = Wav2Vec2Model(config)
# print(model)
# exit()
# model.eval()
# config.encoder_layers = 18
# new_model = Wav2Vec2Model(config)
# model_dict = model.state_dict()
# new_model.load_state_dict(model_dict, strict=False)
# print(new_model.config)

'''toml讀寫'''
# import argparse
# import toml

# if __name__ == '__main__':
#     args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
#     args.add_argument('-c', '--config', required=True, type=str,
#                       help='config file path (default: None)')
#     args = args.parse_args()
#     config = toml.load(args.config)

#     config['dataset'] = {"train_size": "100", "val_size": "20"}
#     print(config)
#     with open('test.toml', 'w+') as f:
#         toml.dump(config, f)
#     f.close()

'''dataloader測試'''
# from utils.utils import *
# from torch.utils.data import DataLoader
# import argparse
# import toml
# from dataloader.dataset import DefaultCollate
# from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
# import torch.distributed as dist
# import datetime
# from logger.pbar import PBar

# if __name__ == '__main__':
#     args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
#     args.add_argument('-c', '--config', required=True, type=str,
#                       help='config file path (default: None)')
#     args = args.parse_args()
#     config = toml.load(args.config)
#     rank = 0
#     stateful_metrics = ["train_loss", "train_lr", "train_grad_norm", "train_wer", "val_loss", "val_wer"]
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '4444'
#     dist.init_process_group("gloo", rank=0, world_size=1, timeout=datetime.timedelta(seconds=3600 * 5))
#     pretrained_path = config['meta']['pretrained_path']
#     config['train_dataset']['args']['sr'] = config['meta']['sr']
#     config['train_dataset']['args']['rank'] = rank
#     config["train_dataset"]["args"]["dist"] = dist
#     config["train_dataset"]["args"]["special_tokens"] = config["special_tokens"]
#     train_base_ds = initialize_module(config["train_dataset"]["path"], args=config["train_dataset"]["args"])
#     tokenizer = Wav2Vec2CTCTokenizer("vocab.json", 
#                                 **config["special_tokens"],
#                                 word_delimiter_token="|")
#     feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_path)
#     processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
#     default_collate = DefaultCollate(processor, config['meta']['sr'])
#     train_ds = train_base_ds.get_data()
#     train_dl = DataLoader(
#         dataset=train_ds,
#         **config["train_dataset"]["dataloader"],
#         collate_fn=default_collate
#     )
#     for dl_step, batch in enumerate(train_dl):
#         pbar = PBar(steps_per_epoch, 10, stateful_metrics = stateful_metrics)

#         print(dl_step, **batch)

'''音檔測試'''
# path = '../../../../../extra_space1/data/audio_data/data_6122/audio'
# total_time, count = 0, 0
# data_list = os.listdir(path)
# for i in tqdm(range(0, len(data_list))):
#     wav, sr = librosa.load(path + f'/{data_list[i]}', sr = 16000)
#     duration = librosa.get_duration(y=wav, sr=sr)

'''libriLight資料處理'''
import argparse
import glob
import os
import random
import soundfile


def libri_data_process(src_path, des_path, ext, copy_file = False):
    count = 1
    audios = []
    labels = []
    # flac_search_path = os.path.join(src_path, "1h/0/**/*." + ext) #libright10m的路徑
    flac_search_path = os.path.join(src_path, "**/*." + ext) #dev_other的路徑
    for fname in sorted(glob.glob(flac_search_path, recursive=True)):
        try:
            if copy_file:
                if not os.path.exists(des_path+'/audio/'):
                    os.makedirs(des_path+'/audio/')
                shutil.copyfile(fname, des_path+'/audio/'+str(count).zfill(5) + '.' + ext)
            temp = fname.split('/')
            file_path = '/'.join(temp[:-1])
            file_name = temp[-1].split('.')[0]
            label_name = '-'.join(file_name.split('-')[:2]) + '.trans.txt'
            # print(fname, temp, file_path, file_name, label_name, sep='\n')
            with open(os.path.join(file_path, label_name), "r") as fp:
                label_data = fp.readlines()
            for e in label_data:
                if file_name in e:
                    label = e.split(file_name)[-1].strip()
            audios.append(str(count).zfill(5) + '.' + ext)
            labels.append(label)
            count += 1
        except:
            print('error', fname)
            continue
    # print(audios, labels, len(audios), len(labels))
    if copy_file:
        write_data(audios, labels, des_path)
        # split_data(des_path)

# src_path = '../../../../../extra_space1/data/libriLight/librispeech_finetuning'
# des_path = '../../../../../extra_space1/data/audio_data/libriLight/10m'
src_path = '../../../../../extra_space1/data/librispeech/dev-other/LibriSpeech/dev-other'
des_path = '../../../../../extra_space1/data/audio_data/librispeech/dev_other'
ext = 'flac'
libri_data_process(src_path, des_path, ext, True)