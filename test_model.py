import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

import Config as config
from Load_Dataset import ValGenerator, ImageToImage2D
from nets.LViT import LViT
from utils import read_text
from sklearn.metrics import jaccard_score

# =====================
# DEVICE SAFE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def compute_metrics(pred, gt):
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    dice = 2 * np.sum(gt * pred) / (np.sum(gt) + np.sum(pred) + 1e-5)
    iou = jaccard_score(gt.reshape(-1), pred.reshape(-1))

    return dice, iou


if __name__ == "__main__":

    model_type = config.model_name
    test_session = config.test_session

    if config.task_name == "MonuSeg":
        model_path = f"./Monuseg/{model_type}/{test_session}/models/best_model-{model_type}.pth.tar"
    else:
        model_path = f"./Covid19/{model_type}/{test_session}/models/best_model-{model_type}.pth.tar"

    # =====================
    # Load Model
    # =====================
    checkpoint = torch.load(model_path, map_location=device)

    config_vit = config.get_CTranS_config()
    model = LViT(config_vit,
                 n_channels=config.n_channels,
                 n_classes=config.n_labels)

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    print("Model loaded!")

    # =====================
    # Load Dataset
    # =====================
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_dataset + 'Test_text.xlsx')

    test_dataset = ImageToImage2D(
        config.test_dataset,
        config.task_name,
        test_text,
        tf_test,
        image_size=config.img_size
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # =====================
    # Threshold Sweep Setup
    # =====================
    thresholds = np.arange(0.1, 0.91, 0.05)
    dice_per_threshold = {t: 0.0 for t in thresholds}
    iou_per_threshold = {t: 0.0 for t in thresholds}
    count = 0

    # =====================
    # Evaluation Loop
    # =====================
    with tqdm(total=len(test_loader), desc="Testing") as pbar:
        for sampled_batch, names in test_loader:

            image = sampled_batch['image'].to(device)
            label = sampled_batch['label']
            text = sampled_batch['text'].to(device)

            with torch.no_grad():
                output = model(image, text)

            prob_map = output[0].detach().cpu().numpy()
            prob_map = np.reshape(prob_map, (config.img_size, config.img_size))
            gt = label.detach().cpu().numpy()[0]

            for thresh in thresholds:
                pred = (prob_map > thresh).astype(np.uint8)
                dice, iou = compute_metrics(pred, gt)

                dice_per_threshold[thresh] += dice
                iou_per_threshold[thresh] += iou

            count += 1
            pbar.update()

    # =====================
    # Threshold Results
    # =====================
    print("\n==================== Threshold Analysis ====================")

    avg_dice_list = []
    avg_iou_list = []

    for t in thresholds:
        avg_dice = dice_per_threshold[t] / count
        avg_iou = iou_per_threshold[t] / count

        avg_dice_list.append(avg_dice)
        avg_iou_list.append(avg_iou)

        print(f"Threshold {t:.2f} → Avg Dice: {avg_dice:.4f} | Avg IoU: {avg_iou:.4f}")

    # Find Best Threshold
    best_index = np.argmax(avg_dice_list)
    best_threshold = thresholds[best_index]
    best_dice = avg_dice_list[best_index]
    best_iou = avg_iou_list[best_index]
    print("\nBest Threshold:", best_threshold)
    print("Best Dice:", best_dice)
    print("Best IOU:", best_iou)

    # =====================
    # Plot Dice Curve
    # =====================f
    plt.figure()
    plt.plot(thresholds, avg_dice_list, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("Average Dice")
    plt.title("Threshold vs Dice")
    plt.grid()
    plt.show()

    # =====================
    # Plot IoU Curve
    # =====================
    plt.figure()
    plt.plot(thresholds, avg_iou_list, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("Average IoU")
    plt.title("Threshold vs IoU")
    plt.grid()
    plt.show()

    # =====================
    # Final Evaluation Using Fixed Threshold (Optional)
    # =====================
    final_threshold = 0.55

    print("\nEvaluating using fixed threshold:", final_threshold)

