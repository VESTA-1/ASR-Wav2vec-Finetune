# import sounddevice as sd
# import torch
# import torchaudio
# from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
# import argparse
# import toml
# import numpy as np
# import keyboard


# def load_asr_model(model_path, device, huggingface_folder):
#     pretrained_path = config['meta']['pretrained_path']
#     tokenizer = Wav2Vec2CTCTokenizer("vocab.json", 
#                                     **config["special_tokens"],
#                                     word_delimiter_token="|")
#     feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_path)
#     processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
#     model = Wav2Vec2ForCTC.from_pretrained(huggingface_folder).to(device)
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint["model"], strict=True)
#     return model, processor


# def asr_inference(model, processor, audio_input):
#     input_values = processor(audio_input, return_tensors="pt", padding="longest", sampling_rate=16000).input_values.to(device)
#     audio_input = audio_input.to(model.device)
#     with torch.no_grad():
#         logits = model(input_values).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)[0]
#     return transcription

# # 修改 callback 函数
# def callback(indata, frames, time, status):
#     if status:
#         print(f'aaaaa{status}', flush=True)
#     audio_input = torch.from_numpy(indata).float().squeeze().to(device)
#     transcription = asr_inference(asr_model, asr_processor, audio_input)
#     print(f"Transcription: {transcription}", flush=True)
#     if keyboard.is_pressed('q'):
#         exit()
#         raise sd.CallbackStop
#         exit()


# if __name__ == "__main__":
#     # 替换成你自己训练好的模型的路径
#     model_path = "./saved/ASR/2023_11_07_16_18_31/checkpoints/best_model.tar"
#     args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
#     args.add_argument('-c', '--config', required=True, type=str,
#                       help='config file path (default: None)')
#     args.add_argument('-r', '--resume', action="store_true",
#                       help='path to latest checkpoint (default: None)')
#     args.add_argument('-p', '--preload', default=None, type=str,
#                       help='Path to pretrained Model')            
#     args.add_argument('-d', '--device_id', type=int, default = 0,
#                       help='The device you want to test your model on if CUDA is available. Otherwise, CPU is used. Default value: 0')
#     args.add_argument('-s', '--huggingface_folder', type=str, default = 'huggingface-hub',
#                       help='The folder where you stored the huggingface files. Check the <local_dir> argument of [huggingface.args] in config.toml. Default value: "huggingface-hub".')
#     args = args.parse_args()
#     config = toml.load(args.config)
#     print(config["special_tokens"])
#     device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

#     # 加载模型
#     asr_model, asr_processor = load_asr_model(model_path, device, huggingface_folder = args.huggingface_folder)

#     # 设置音频参数
#     samplerate = asr_processor.feature_extractor.sampling_rate
#     channels = 1  # 单声道
#     dtype = np.float32

#     # 实时录制音频
#     with sd.InputStream(callback=callback, channels=channels, dtype=dtype, samplerate=samplerate):
#         sd.sleep(1000000)


import librosa
import numpy as np
import pyaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment


model_path = "./saved/ASR/2023_11_20_12_02_40/checkpoints/best_model.tar"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


pretrained_path = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained('huggingface-hub')
model = Wav2Vec2ForCTC.from_pretrained('huggingface-hub').to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model"], strict=True)

samplerate = processor.feature_extractor.sampling_rate

r = sr.Recognizer()

with sr.Microphone(sample_rate=16000) as source:
    print('開始錄音')
    while True:
        audio = r.listen(source, timeout=5)
        data = io.BytesIO(audio.get_wav_data())
        clip = AudioSegment.from_file(data)
        x = torch.FloatTensor(clip.get_array_of_samples())

        input_values = processor(x, sampling_rate=16000, return_tensors="pt", padding='longest').input_values
        print("處理中")
        logits = model(input_values.to(device)).logits
        print("處理中2")
        pred_ids = torch.argmax(logits, dim=-1)
        print("處理中3")
        pred_transcript = processor.batch_decode(pred_ids)[0]

        print(f"識別結果: {pred_transcript}")