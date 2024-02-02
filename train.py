import argparse
import json
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import toml
import warnings
import datetime
import time
warnings.filterwarnings('ignore')

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from time import gmtime, strftime
from utils.utils import *
from utils.metric import Metric
from dataloader.dataset import DefaultCollate
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4444'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600 * 5))

def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, config, resume, preload):
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES']=config["meta"]["device_ids"]
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    setup(rank, world_size)
    # print(f'-------------------{rank} {world_size}')
    epochs = config["meta"]["epochs"]
    gradient_accumulation_steps = config["meta"]["gradient_accumulation_steps"]
    use_amp = config["meta"]["use_amp"]
    max_clip_grad_norm = config["meta"]["max_clip_grad_norm"]
    now = datetime.datetime.now()
    save_dir =  os.path.join(config["meta"]["save_dir"], config["meta"]['name'], now.strftime("%Y_%m_%d_%H_%M_%S"), 'checkpoints')
    log_dir = os.path.join(config["meta"]["save_dir"], config["meta"]['name'], now.strftime("%Y_%m_%d_%H_%M_%S"), 'log_dir')
    if rank == 0:
        # Creatr dirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Store config file
        config_name = strftime("%Y-%m-%d %H_%M_%S", gmtime()).replace(' ', '_') + '.toml'
        with open(os.path.join(config["meta"]["save_dir"], config["meta"]['name'], now.strftime("%Y_%m_%d_%H_%M_%S") + '/' + config_name), 'w+') as f:
            toml.dump(config, f)
            f.close()

    # This should be needed to be reproducible https://discuss.pytorch.org/t/setting-seed-in-torch-ddp/126638
    config["meta"]["seed"] += rank
    set_seed(config["meta"]["seed"])
    config['val_dataset']['args']['sr'] = config['meta']['sr']
    config['train_dataset']['args']['sr'] = config['meta']['sr']

    config['train_dataset']['args']['rank'] = rank
    config['val_dataset']['args']['rank'] = rank

    config["train_dataset"]["args"]["dist"] = dist
    config["val_dataset"]["args"]["dist"] = dist

    config["train_dataset"]["args"]["special_tokens"] = config["special_tokens"]
    config["val_dataset"]["args"]["special_tokens"] = config["special_tokens"]
    pretrained_path = config['meta']['pretrained_path']
    train_base_ds = initialize_module(config["train_dataset"]["path"], args=config["train_dataset"]["args"])
    vocab_dict = train_base_ds.get_vocab_dict()
    with open('vocab.json', 'w+', encoding='utf8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
        f.close()
    dist.barrier()
    # Create processor
    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", 
                                    **config["special_tokens"],
                                    word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_path)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    default_collate = DefaultCollate(processor, config['meta']['sr'])

    # Create train dataloader
    train_ds = train_base_ds.get_data()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        **config["train_dataset"]["sampler"]
    )
    train_dl = DataLoader(
        dataset=train_ds,
        **config["train_dataset"]["dataloader"],
        sampler = train_sampler,
        collate_fn=default_collate
    )

    # Create val dataloader
    val_base_ds = initialize_module(config["val_dataset"]["path"], args=config["val_dataset"]["args"])
    val_ds = val_base_ds.get_data()
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        **config["val_dataset"]["sampler"]
    )
    val_dl = DataLoader(
        dataset=val_ds,
        **config["val_dataset"]["dataloader"],
        sampler = val_sampler,
        collate_fn=default_collate
    )


    # Load pretrained model
    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_path, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        gradient_checkpointing=False
    )
    
    # freeze the wav2vec feature encoder, if you have small dataset, this helps a lot
    model.freeze_feature_encoder()
    # DDP for multi-processing
    model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=True)

    # Set up metric, scheduler, optmizer
    compute_metric = Metric(processor)

    # optimizer = torch.optim.AdamW(
    #     params = model.parameters(),
    #     lr = config["optimizer"]["lr"]
    # )

    # optimizer = torch.optim.Adadelta(
    #     params = model.parameters(),
    #     lr = config["optimizer"]["lr"]
    # )

    # optimizer = torch.optim.SGD(
    #     params = model.parameters(),
    #     lr = config["optimizer"]["lr"]
    # )

    optimizer_class = initialize_module(config["optimizer"]["path"], initialize=False)
    optimizer = optimizer_class(
        params = model.parameters(),
        lr = config["optimizer"]["lr"]
    )

    print(optimizer)
    steps_per_epoch = (len(train_dl)//gradient_accumulation_steps) + (len(train_dl)%gradient_accumulation_steps != 0)

    scheduler_class = initialize_module(config["scheduler"]["path"], initialize=False)
    scheduler = scheduler_class(
        optimizer,
        max_lr=config["scheduler"]["max_lr"], 
        epochs=epochs,
        steps_per_epoch = steps_per_epoch)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    # optimizer, 
    # max_lr=config["scheduler"]["max_lr"], 
    # epochs=epochs, 
    # steps_per_epoch = steps_per_epoch)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, 
    #     max_lr=config["scheduler"]["max_lr"], 
    #     epochs=epochs, 
    #     steps_per_epoch = steps_per_epoch,
    #     three_phase=False,
    #     cycle_momentum=False)
    
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, 
    #     max_lr=config["scheduler"]["max_lr"], 
    #     epochs=epochs, 
    #     steps_per_epoch = steps_per_epoch,
    #     three_phase=True)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["scheduler"]["max_lr"], max_lr=4e-6, cycle_momentum=False)

    if rank == 0:
        print("Number of training utterances: ", len(train_ds))
        print("Number of validation utterances: ", len(val_ds))

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)
    trainer = trainer_class(
        dist = dist,
        rank = rank,
        n_gpus = world_size,
        config = config,
        resume = resume,
        preload = preload,
        epochs = epochs,
        steps_per_epoch = steps_per_epoch,
        model = model,
        compute_metric = compute_metric,
        processor = processor,
        train_dl = train_dl,
        val_dl = val_dl,
        train_sampler = train_sampler,
        val_sampler = val_sampler,
        optimizer = optimizer,
        scheduler = scheduler,
        save_dir = save_dir,
        log_dir = log_dir,
        gradient_accumulation_steps = gradient_accumulation_steps,
        use_amp = use_amp,
        max_clip_grad_norm = max_clip_grad_norm
    )
    trainer.train()


    cleanup()
    end_time = time.time()
    def seconds_to_hours(seconds):
        hours = seconds // 3600
        remaining_seconds = seconds % 3600
        minutes = remaining_seconds // 60
        remaining_seconds = remaining_seconds % 60
    
        return hours, minutes, remaining_seconds
    total_time = end_time - start_time
    hours, minutes, remaining_seconds = seconds_to_hours(total_time)
    print(f"總執行時間：{hours} hr, {minutes} min, {remaining_seconds} sec")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', action="store_true",
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-p', '--preload', default=None, type=str,
                      help='Path to pretrained Model')            
    
    args = args.parse_args()
    config = toml.load(args.config)
    n_gpus = len(config['meta']["device_ids"].split(','))
    
    mp.spawn(
        main,
        args = (n_gpus, config, args.resume, args.preload),
        nprocs = n_gpus,
        join = True
    )

