import torch.optim as optim
import os
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from .utils import EarlyStopping


def trainModel(
    model,
    history,
    train_loader,
    val_loader,
    model_type,
    save_path="/content/drive/MyDrive/regularization-data/weights",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_val_acc = 0.0

    # Training loop
    max_epochs = 70

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Add early stopping
    if not os.path.isdir(save_path):
        print(
            f"The data storage directory: {save_path},  does not exist."
            "/nCreating the folder to store weights and model metadata"
        )
        os.mkdir(save_path)
        print(f"Folder {save_path} created successfully.")

    early_stopping = EarlyStopping(
        patience=10, delta=0.001, path=f"{save_path}/{model_type}.pth"
    )
    # Early stopping object saves model based on best val acc.

    for epoch in range(max_epochs):
        print("-" * 50)
        print(f"Epoch {epoch+1}/{max_epochs}")
        start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # tqdm for progress bar
        train_pbar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)

            # Update progress bar every 10 batches
            if batch_idx % 10 == 0:
                train_pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "batch": f"{batch_idx}/{len(train_loader)}",
                    }
                )

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        val_pbar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.float() / len(val_loader.dataset)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Update scheduler
        scheduler.step(val_acc)

        # Print and save the current LR
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr}")

        early_stopping(val_loss, model)

        # Save model at last epoch
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        # Print progress
        epoch_time = time.time() - start_time

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(
            f'Best Val Acc: {best_val_acc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )
        print(f"Time: {epoch_time:.1f}s")
        print("-" * 50)
        print("\n")

    print(f"Training complete.")
    return model
