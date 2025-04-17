import os
import torch
import numpy as np
from glob import glob
from torch import optim
from torch.utils.data import DataLoader
# from torchmetrics.classification import Precision, Recall, IoU
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Precision, Recall, JaccardIndex


from data_generator import KvasirSegDataset  # Custom Data Generator for PyTorch
from resunetplusplus import build_resunetplusplus  # Updated model import for PyTorch
from metrics import dice_coef, dice_loss  # Custom metrics in PyTorch

if __name__ == "__main__":
    ## Path
    file_path = "../drive/MyDrive/polyp_segmentation/ResUnetPlusPlus/files/"
    # model_path = "../drive/MyDrive/polyp_segmentation/ResUnetPlusPlus/files/resunetplusplus.pth"

    periodic_model_dir = os.path.join(file_path, "checkpoints")
    os.makedirs(periodic_model_dir, exist_ok=True)

    ## Create files folder if not exists
    os.makedirs(file_path, exist_ok=True)

    train_path = "new_data/polyp-dataset/train/"
    valid_path = "new_data/polyp-dataset/valid/"

    ## Training data
    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()

    ## Validation data
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()

    ## Parameters
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 200

    train_steps = len(train_image_paths) // batch_size
    valid_steps = len(valid_image_paths) // batch_size

    ## DataLoader (converted from DataGen)
    train_dataset = KvasirSegDataset(train_image_paths, train_mask_paths, image_size=image_size)
    valid_dataset = KvasirSegDataset(valid_image_paths, valid_mask_paths, image_size=image_size)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    ## ResUnet++
    model = build_resunetplusplus()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    ## Optimizer and loss function
    optimizer = optim.NAdam(model.parameters(), lr=lr)
    # precision_metric = Precision()
    # recall_metric = Recall()
    # iou_metric = IoU(num_classes=2)
    # precision_metric = Precision(task="binary").to('cuda' if torch.cuda.is_available() else 'cpu')
    # recall_metric = Recall(task="binary").to('cuda' if torch.cuda.is_available() else 'cpu')
    # iou_metric = JaccardIndex(task="binary").to('cuda' if torch.cuda.is_available() else 'cpu')

    ## TensorBoard
    writer = SummaryWriter(file_path)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for images, masks in train_loader:
            images, masks = images.to('cuda' if torch.cuda.is_available() else 'cpu'), masks.to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(masks, outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / train_steps
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        ## Validation loop
        model.eval()
        running_val_loss = 0.0

        # precision_metric.reset()
        # recall_metric.reset()
        # iou_metric.reset()

        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to('cuda' if torch.cuda.is_available() else 'cpu'), masks.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = model(images)
                loss = dice_loss(masks, outputs)
                running_val_loss += loss.item()

                preds = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
                preds = (preds > 0.5).float()  # Convert to binary predictions (0 or 1)

                # Update metrics
                # precision_metric.update(preds, masks.int())  # Update precision metric
                # recall_metric.update(preds, masks.int())  # Update recall metric
                # iou_metric.update(preds, masks.int()) 


        avg_val_loss = running_val_loss / valid_steps
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        # precision = precision_metric.compute().item()  # Compute precision
        # recall = recall_metric.compute().item()  # Compute recall
        # iou = iou_metric.compute().item()  # Compute IoU

        # Log to TensorBoard
        # writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Metrics/precision', precision, epoch)
        writer.add_scalar('Metrics/recall', recall, epoch)
        writer.add_scalar('Metrics/iou', iou, epoch)

        # Print metrics
        # print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
        #     f"Precision: {precision:.4f} | Recall: {recall:.4f} | IoU: {iou:.4f}")

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        ## Save model if improved
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), model_path)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(file_path, "resunetplusplus_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model at epoch {epoch + 1}")

        # Save model every 10 epochs
        if (epoch + 1) % 5 == 0:
            epoch_model_path = os.path.join(periodic_model_dir, f"resunetplusplus_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            print(f"Saved model at epoch {epoch + 1}")

        ## Learning rate scheduler (if needed)
        if epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] = max(optimizer.param_groups[0]['lr'] * 0.1, 1e-6)

    print("Training completed.")
    writer.close()
