from datasets import load_metric
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
class Metric:
    def __init__(self, processor):
        self.processor = processor
        self.wer_metric = load_metric("wer")
    def __call__(self, logits, labels):
        preds = torch.argmax(logits, axis=-1)
        labels[labels == -100] = self.processor.tokenizer.pad_token_id

        pred_strs = self.processor.batch_decode(preds)
        # we do not want to group tokens when computing the metrics
        label_strs = self.processor.batch_decode(labels, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_strs, references=label_strs)
        return wer


if __name__ == '__main__':
    processor = Wav2Vec2Processor.from_pretrained('huggingface-hub')
    compute_metric = Metric(processor)
            # wer = torch.tensor(compute_metric(outputs.logits.detach(), batch['labels']))
    print('a')