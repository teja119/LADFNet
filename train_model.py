# -*- coding: utf-8 -*-
import torch
import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
import Config
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D, LV2D
from nets.LViT import LViT
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch, print_summary
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch
from thop import profile

# ============================
# DEVICE (SAFE FOR CPU/GPU)
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']
    best_model = state['best_model']
    model = state['model']

    if best_model:
        filename = save_path + '/' + f'best_model-{model}.pth.tar'
    else:
        filename = save_path + '/' + f'model-{model}-{epoch:02d}.pth.tar'

    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):

    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])

    if config.task_name == 'MonuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')

        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf,
                                     image_size=config.img_size)

    elif config.task_name == 'Covid19':
        text = read_text(config.task_dataset + 'Train_Val_text.xlsx')

        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, text, val_tf,
                                     image_size=config.img_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=4,
                              pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=4,
                            pin_memory=torch.cuda.is_available())

    lr = config.learning_rate
    logger.info(model_type)

    if model_type in ['LViT', 'LViT_pretrain']:

        config_vit = config.get_CTranS_config()
        logger.info(f'transformer head num: {config_vit.transformer.num_heads}')
        logger.info(f'transformer layers num: {config_vit.transformer.num_layers}')
        logger.info(f'transformer expand ratio: {config_vit.expand_ratio}')

        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

        if model_type == 'LViT_pretrain':
            pretrained_path = "MoNuSeg/LViT/Test_session_05.23_10h55/models/best_model-LViT.pth.tar"
            pretrained = torch.load(pretrained_path, map_location=device)
            pretrained = pretrained['state_dict']
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in pretrained.items() if k in model_dict}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            logger.info('Load successful!')

    else:
        raise TypeError('Please enter a valid name for the model type')

    # FLOPS CHECK
    input = torch.randn(2, 3, 224, 224)
    text = torch.randn(2, 10, 768)
    flops, params = profile(model, inputs=(input, text))
    print(f'flops:{flops}')
    print(f'params:{params}')

    # 🔥 SAFE DEVICE MOVE
    model = model.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if config.cosineLR:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None

    writer = SummaryWriter(config.tensorboard_folder) if tensorboard else None

    max_dice = 0.0
    best_epoch = 1

    for epoch in range(config.epochs):

        logger.info(f'\n========= Epoch [{epoch + 1}/{config.epochs}] =========')

        model.train(True)
        train_one_epoch(train_loader, model, criterion, optimizer, writer,
                        epoch, None, model_type, logger)

        logger.info('Validation')

        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                 optimizer, writer, epoch,
                                                 lr_scheduler, model_type, logger)

        if val_dice > max_dice and epoch + 1 > 5:
            logger.info(f'\t Saving best model: {val_dice:.4f}')
            max_dice = val_dice
            best_epoch = epoch + 1

            save_checkpoint({
                'epoch': epoch,
                'best_model': True,
                'model': model_type,
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict()
            }, config.model_path)

        if epoch - best_epoch + 1 > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':

    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)

    model = main_loop(model_type=config.model_name, tensorboard=True)