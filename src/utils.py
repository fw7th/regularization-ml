import torch
import json
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from thop import profile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"


# Implementing an early stopping class based on val loss
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss  # lower loss = better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            print("💾 Saved best model!")

    def save_checkpoint(self, val_loss, model):
        if not os.path.isdir(self.path):
            print(
                f"The data storage directory: {self.path},  does not exist.\n"
                "Creating the folder to store weights and model metadata"
            )
            os.mkdir(self.path)
            print(f"Folder {self.path} created successfully.")

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def loadWeights(model, weights_path):
    if not os.path.exists(weights_path):
        print("Model weights not found.")
    else:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    return model


def readJson(file_path):
    if not os.path.exists(file_path):
        print("Json file not found.")
    else:
        with open(file_path, "r") as f:
            data = json.load(f)
    return data


def genError(save_path, train_losses, val_losses):
    # This function plots generalization gap across epochs using losses
    gen_err_list = []
    for epoch in range(len(train_losses)):
        train_loss = train_losses[epoch]
        val_loss = val_losses[epoch]
        generalization_error = val_loss - train_loss
        gen_err_list.append(generalization_error)

    epochs = range(1, len(gen_err_list) + 1)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(epochs, gen_err_list, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss - Training Loss")
    plt.title("Generalization Gap Across Epochs")
    plt.grid(True)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if not os.path.exists(save_path):
        print(f"Save path not found, creating and saving to path...")
        os.mkdir(save_path)

    plt.savefig(
        f"{save_path}/genErrorPlot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


def saveHistory(history, save_path):
    if not os.path.exists(save_path):
        print("Save path does not exist, creating..")
        os.makedirs(save_path)

    # Convert any tensors to regular Python numbers
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, list):
            history_serializable[key] = [
                item.item() if hasattr(item, "item") else item for item in value
            ]
        else:
            history_serializable[key] = value

    with open(save_path, "w") as f:
        json.dump(history_serializable, f, indent=4)

    print("File Saved!")


def evalModel(model, visualizations_path, test_loader):
    y_true = []  # ground truth labels
    y_pred = []  # predicted labels

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            all_labels.append(labels)
            all_preds.append(preds)

    # concat all batches, we vectorize
    y_true = torch.cat(all_labels).cpu().numpy()
    y_pred = torch.cat(all_preds).cpu().numpy()

    # Accuracy
    print("Accuracy:", accuracy_score(y_true, y_pred))

    # macro averages included in the report
    print(classification_report(y_true, y_pred, digits=4))

    # FLOPs and Param count
    model_copy = copy.deepcopy(model)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model_copy, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}, Number of Params: {params:,}\n")

    # Confustion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if not os.path.exists(visualizations_path):
        print(f"visualizations path not found, creating and saving to path...")
        os.mkdir(visualizations_path)

    plt.savefig(
        f"{visualizations_path}/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)
