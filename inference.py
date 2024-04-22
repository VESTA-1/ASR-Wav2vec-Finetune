import librosa
import torch
import os
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from tqdm import tqdm
from datasets import load_metric

class Inferencer:
    def __init__(self, device, huggingface_folder, model_path) -> None:
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(huggingface_folder)
        self.model = Wav2Vec2ForCTC.from_pretrained(huggingface_folder).to(self.device)
        self.wer_metric = load_metric("wer")
        if model_path is not None:
            self.preload_model(model_path)


    def preload_model(self, model_path) -> None:
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.
        Args:
            model_path: The file path of the *.tar file
        """
        assert os.path.exists(model_path), f"The file {model_path} is not exist. please check path."
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict = True)
        print(f"Model preloaded successfully from {model_path}.")


    def transcribe(self, wav) -> str:
        input_values = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_values
        logits = self.model(input_values.to(self.device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_transcript = self.processor.batch_decode(pred_ids)[0]
        return pred_transcript

    def run(self, test_filepath, model_path):
        wer = 0
        model_name = model_path.split('/')[-3]
        temp = test_filepath.split('/')[-1].split('.')
        data_name = test_filepath.split('/')[-2]
        filename = temp[0]
        filetype = temp[1]
        if filetype == 'txt':
            f = open(test_filepath, 'r', encoding='utf8')
            lines = f.read().splitlines()
            f.close()
            lines.pop(0)
            if not os.path.isdir(f'./result/{model_name}/{data_name}'):
                os.makedirs(f'./result/{model_name}/{data_name}')
            f = open(f'./result/{model_name}/{data_name}/transcript_{filename}.txt', 'w+', encoding='utf8')
            for line in tqdm(lines):
                data, label = line.split('|')
                wav, _ = librosa.load(data, sr = 16000)
                transcript = self.transcribe(wav)
                wer += torch.tensor(self.evaluate_wer(transcript, label))
                f.write(line + '|' + transcript + '\n')
            print(wer)
            wer /= len(lines)
            print(wer)
            f.write("wer ： " + str(wer))
            f.close()
        else:
            wav, _ = librosa.load(test_filepath, sr = 16000)
            print(f"transcript: {self.transcribe(wav)}")

    def evaluate_wer(self, pred, label):
        # pred_strs = self.processor.batch_decode(pred)
        # label_strs = self.processor.batch_decode(label, group_tokens=False)
        # wer = self.wer_metric.compute(predictions=pred_strs, references=label_strs)
        wer = self.wer_metric.compute(predictions=[pred], references=[label])
        return wer

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR INFERENCE ARGS')
    args.add_argument('-f', '--test_filepath', type=str, required = True,
                      help='It can be either the path to your audio file (.wav, .mp3) or a text file (.txt) containing a list of audio file paths.')
    args.add_argument('-s', '--huggingface_folder', type=str, default = 'huggingface-hub',
                      help='The folder where you stored the huggingface files. Check the <local_dir> argument of [huggingface.args] in config.toml. Default value: "huggingface-hub".')
    args.add_argument('-m', '--model_path', type=str, default = None,
                      help='Path to the model (.tar file) in saved/<project_name>/checkpoints. If not provided, default uses the pytorch_model.bin in the <HUGGINGFACE_FOLDER>')
    args.add_argument('-d', '--device_id', type=int, default = 0,
                      help='The device you want to test your model on if CUDA is available. Otherwise, CPU is used. Default value: 0')
    args = args.parse_args()
    
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    inferencer = Inferencer(
        device = device, 
        huggingface_folder = args.huggingface_folder, 
        model_path = args.model_path)

    inferencer.run(args.test_filepath, args.model_path)

