import torch
import json
import copy
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from thop import profile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def unpickle(file):
    import pickle
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

def loadWeights(model, model_type, model_weights):
    PATH = f"{base}/{model_type}/{model_weights}.pth"
    state_dict = torch.load(PATH, map_location=device)
    model.load_state_dict(state_dict)

    return model

def readJson(model_type):
    json_file = f'{base}/{model_type}/history_{model_type}.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def genError(train_losses, val_losses, model_type, model_epoch):
    # This function plots generalization gap across epochs using losses
    gen_err_list = []
    for epoch in range(len(train_losses)):
        train_loss = train_losses[epoch]
        val_loss = val_losses[epoch]
        generalization_error = val_loss - train_loss
        gen_err_list.append(generalization_error)

    epochs = range(1, len(gen_err_list) + 1)
    fig = plt.figure(figsize=(8,6))
    plt.plot(epochs, gen_err_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss - Training Loss')
    plt.title('Generalization Gap Across Epochs')
    plt.grid(True)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f"{base}/{model_type}/generalization_error_{model_epoch}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
def saveHistory(history, model_type):
    # Convert any tensors to regular Python numbers
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, list):
            history_serializable[key] = [
                item.item() if hasattr(item, 'item') else item
                for item in value
            ]
        else:
            history_serializable[key] = value

    with open(f"{base}/{model_type}/history_{model_type}.json", "w") as f:
        json.dump(history_serializable, f, indent=4)

    print("File Saved!")

def evalModel(model, model_type, model_weights):
    model_to_eval = loadWeights(model, model_type, model_weights)

    y_true = [] # ground truth labels
    y_pred = [] # predicted labels

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model_to_eval(inputs)
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
    model_to_eval_copy = copy.deepcopy(model_to_eval)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model_to_eval_copy, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}, Number of Params: {params:,}\n")

    # Confustion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f"{base}/{model_type}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
