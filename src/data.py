from .utils import unpickle
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom pytorch dataset.
    """

    def __init__(self, data, labels, transform=None):
        self.data = data.reshape(-1, 3, 32, 32)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.labels[idx]
        sample = Image.fromarray(sample.transpose(1, 2, 0))  # HWC
        if self.transform:
            sample = self.transform(sample)

        return sample, label


# Example usage:
# Assuming 'my_data' and 'my_labels' are lists or arrays of your data
# dataset = CustomDataset(my_data, my_labels, transform=my_transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


class Cutout(object):
    """Randomly mask out one or more square regions from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of cutout applied.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def Loader(
    train_data,
    train_labels,
    val_data,
    val_labels,
    test_data,
    test_labels,
    cutout=None,
    cutout_size=None,
):
    """
    Function returns the train, test and val loaders, after transformations.
    """
    print("Starting Data Loading...")

    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(
            f"   Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB"
        )
    else:
        print("⚠️  CUDA not available, using CPU")

    if (
        cutout and not cutout_size
    ):  # Ensure cutout size is specified when cutout == True
        print("Cutout size must be specified to use cutout")
        return

    transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        ),
    ]

    if cutout and cutout_size:
        transforms_list.append(Cutout(1, cutout_size))

    train_transform = transforms.Compose(transforms_list)
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    print("📁 Loading datasets...")
    try:
        train_dataset = CustomDataset(
            train_data, train_labels, transform=train_transform
        )
        test_dataset = CustomDataset(test_data, test_labels, transform=val_transform)
        val_dataset = CustomDataset(val_data, val_labels, transform=val_transform)

        print("✅ Datasets loaded successfully")
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return None

    # Use smaller batch size for debugging
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")

    # Test data loading first
    if not check_data_loading(train_loader, val_loader):
        print("Data loading test failed!")
        return None

    return train_loader, val_loader, test_loader


def load_cifar_10_data(directory):
    """
    Loads the cifar-10-dataset.

    Args:
        directory (str): The base of the cifar-10 dataset.

    Returns:
        train_data (list): Python list of the training data.
        train_labels (list): Python list of the training labels matching the data.

        // test and val labels and data return similar values //
        // for the testing and validation sets //
    """

    train_data = []
    train_labels = []
    for i in range(1, 6):
        d = unpickle(f"{directory}/cifar-10-batches-py/data_batch_{i}")
        train_data.append(d[b"data"])
        train_labels.extend(d[b"labels"])
    training_data = np.vstack(train_data)
    training_labels = np.array(train_labels)

    test_dict = unpickle(f"{directory}/cifar-10-batches-py/test_batch")
    test_data = test_dict[b"data"]
    test_labels = np.array(test_dict[b"labels"])

    train_data = training_data[10000:]
    train_labels = training_labels[10000:]

    val_data = training_data[:10000]
    val_labels = training_labels[:10000]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def class_to_idx(metadata):
    """
    Custom class to idx implementation for cifar-10 batches.meta file.

    Args:
        metadata: Path to cifar-10 metadata file.
    """
    label_dict = unpickle(metadata)
    labels = [name.decode("utf-8") for name in label_dict[b"label_names"]]
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(labels))}
    print(class_to_idx)


def check_data_loading(train_loader, val_loader):
    """Test data loading to make sure it works"""
    print("🔍 Testing data loading...")

    # Test train loader
    try:
        batch = next(iter(train_loader))
        inputs, labels = batch
        print(f"✅ Train batch shape: {inputs.shape}, Labels: {labels.shape}")
        print(f"   Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"   Label range: [{labels.min()}, {labels.max()}]")
    except Exception as e:
        print(f"❌ Train loader error: {e}")
        return False

    # Test val loader
    try:
        batch = next(iter(val_loader))
        inputs, labels = batch
        print(f"✅ Val batch shape: {inputs.shape}, Labels: {labels.shape}")
    except Exception as e:
        print(f"❌ Val loader error: {e}")
        return False

    return True
