import os
import torch
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

from resunetplusplus import build_resunetplusplus
from data_generator import parse_image, parse_mask
from metrics import dice_coef  # Optional if you want to evaluate

def mask_to_3d(mask):
    """Convert single-channel mask to 3-channel for visualization."""
    mask = np.squeeze(mask)
    mask = np.stack([mask]*3, axis=-1)
    return mask

if __name__ == "__main__":
    file_path = "../drive/MyDrive/polyp_segmentation/ResUnetPlusPlus/files/"
    model_path = os.path.join(file_path, "resunetplusplus_best.pth")
    save_path = os.path.join(file_path, "results")
    test_path = "new_data/polyp-dataset/test/"

    image_size = 256

    test_image_paths = sorted(glob(os.path.join(test_path, "images", "*")))
    test_mask_paths = sorted(glob(os.path.join(test_path, "masks", "*")))

    ## Create result folder
    os.makedirs(save_path, exist_ok=True)

    ## Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resunetplusplus()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Generating test results...")

    dice_scores = []

    for i, (img_path, mask_path) in tqdm(enumerate(zip(test_image_paths, test_mask_paths)), total=len(test_image_paths)):
        ## Load and preprocess image and mask
        image = parse_image(img_path, image_size=image_size)
        mask = parse_mask(mask_path, image_size=image_size)

        input_tensor = torch.from_numpy(image).unsqueeze(0).to(device)  # (1, 3, H, W)
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float().cpu().numpy()[0, 0]

        ## Optional Dice score calculation
        dice = dice_coef(torch.from_numpy(mask), torch.from_numpy(pred_mask)).item()
        dice_scores.append(dice)

        ## Prepare for visualization
        pred_mask_vis = mask_to_3d(pred_mask) * 255
        true_mask_vis = mask_to_3d(mask) * 255
        input_vis = np.transpose(image, (1, 2, 0)) * 255

        sep_line = np.ones((image_size, 10, 3), dtype=np.uint8) * 255
        result_image = np.concatenate([input_vis, sep_line, true_mask_vis, sep_line, pred_mask_vis], axis=1)

        cv2.imwrite(os.path.join(save_path, f"{i}.png"), result_image)

    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice Coefficient on Test Set: {mean_dice:.4f}")
    print("Test image generation complete.")
