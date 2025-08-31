import matplotlib.pyplot as plt
from utils import loadWeights

def plotFmaps_and_activationHist(model, model_type, model_weights, layer=0):
    # Load weights
    model = loadWeights(model, model_type, model_weights)
    model.to(device)
    model.train()  # keep dropout active

    # Grab one sample
    for batch in val_loader:
        imgs, labels = batch
        break

    img0, label0 = imgs[0], labels[0]
    img0 = img0.to(device)

    # hook to capture activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook

    # Register hook on chosen conv layer
    model.conv[layer].block[0].register_forward_hook(get_activation(f"conv{layer}"))

    # Forward pass
    img0 = img0.unsqueeze(0)
    _ = model(img0)   # forward through model with dropout ON

    # Get the feature maps
    feat_maps = activations[f"conv{layer}"].squeeze(0)  # shape: (channels, H, W)

    # --- Feature maps plotting ---
    num_maps = min(32, feat_maps.shape[0])
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    for i in range(num_maps):
        axes[i // 8, i % 8].imshow(feat_maps[i], cmap="gray")
        axes[i // 8, i % 8].axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f"{base}/{model_type}/{model_weights}_fmaps_layer{layer}.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print("\n\n")

    # --- Activation histogram plotting ---
    flat_acts = feat_maps.numpy().ravel()
    plt.figure(figsize=(8,6))
    plt.hist(flat_acts, bins=100, color="blue", alpha=0.7)
    plt.title(f"Activation Histogram - Layer {layer}")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.savefig(f"{base}/{model_type}/{model_weights}_hist_layer{layer}.png",
                dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plotCurves(train_losses, val_losses, train_accs, val_accs, model_type):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(epochs, train_losses, label='Train Loss')
    ax.plot(epochs, val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True)

    fig.savefig(f"{base}/{model_type}/train&val_loss.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("\n\n")

    # Plot Accuracy
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(epochs, train_accs, label='Train Acc')
    ax.plot(epochs, val_accs, label='Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training & Validation Accuracy')
    ax.legend()
    ax.grid(True)

    fig.savefig(f"{base}/{model_type}/train&val_acc.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)