# -*- coding: utf-8 -*-
import torch
import os
import time
from utils import *
import Config as config
import warnings
from torchinfo import summary
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# =========================
# DEVICE SAFE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):

    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)

    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)

    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)

    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)


def train_one_epoch(loader, model, criterion, optimizer, writer,
                    epoch, lr_scheduler, model_type, logger):

    logging_mode = 'Train' if model.training else 'Val'
    end = time.time()

    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum = 0.0, 0.0
    dices = []

    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        images = sampled_batch['image'].to(device)
        masks = sampled_batch['label'].to(device)
        text = sampled_batch['text'].to(device)

        # Ensure correct token length
        if text.dim() == 3 and text.shape[1] > 10:
            text = text[:, :10, :]

        # ===============================
        # Forward
        # ===============================
        preds = model(images, text)
        out_loss = criterion(preds, masks.float())

        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        train_dice = criterion._show_dice(preds, masks.float())
        train_iou = iou_on_batch(masks, preds)

        batch_time = time.time() - end

        # ===============================
        # Visualization (Val only)
        # ===============================
        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path + str(epoch) + '/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images, masks, preds, names, vis_path)

        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss.item()
        iou_sum += len(images) * train_iou
        dice_sum += len(images) * train_dice

        total_samples = (i - 1) * config.batch_size + len(images)

        average_loss = loss_sum / total_samples
        average_time = time_sum / total_samples
        train_iou_average = iou_sum / total_samples
        train_dice_avg = dice_sum / total_samples

        end = time.time()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader),
                          out_loss.item(), loss_name, batch_time,
                          average_loss, average_time,
                          train_iou, train_iou_average,
                          train_dice, train_dice_avg,
                          0, 0, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),
                          logger=logger)

        if config.tensorboard and writer is not None:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        # Only clear cache if GPU exists
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg