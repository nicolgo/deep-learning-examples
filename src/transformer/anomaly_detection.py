import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from transformer import TransformerPredictor, get_pretrained_model

CHECKPOINT_PATH = "../saved_models/tutorial6"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def extract_features(dataset, save_file):
    if not os.path.isfile(save_file):
        data_loader = data.DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
        extracted_features = []
        for imgs, _ in tqdm(data_loader):
            imgs = imgs.to(device)
            feats = pretrained_model(imgs)
            extracted_features.append(feats)
        extracted_features = torch.cat(extracted_features, dim=0)
        extracted_features = extracted_features.detach().cpu()
        torch.save(extracted_features, save_file)
    else:
        extracted_features = torch.load(save_file)
    return extracted_features


class SetAnomalyDataset(data.Dataset):

    def __init__(self, img_feats, labels, set_size=10, train=True):
        """
        Inputs:
            img_feats - Tensor of shape [num_imgs, img_dim]. Represents the high-level features.
            labels - Tensor of shape [num_imgs], containing the class labels for the images
            set_size - Number of elements in a set. N-1 are sampled from one class, and one from another one.
            train - If True, a new set will be sampled every time __getitem__ is called.
        """
        super().__init__()
        self.img_feats = img_feats
        self.labels = labels
        self.set_size = set_size - 1  # The set size is here the size of correct images
        self.train = train

        # Tensors with indices of the images per class
        self.num_labels = labels.max() + 1
        self.img_idx_by_label = torch.argsort(self.labels).reshape(self.num_labels, -1)

        if not train:
            self.test_sets = self._create_test_sets()

    def _create_test_sets(self):
        # Pre-generates the sets for each image for the test set
        test_sets = []
        num_imgs = self.img_feats.shape[0]
        np.random.seed(42)
        test_sets = [self.sample_img_set(self.labels[idx]) for idx in range(num_imgs)]
        test_sets = torch.stack(test_sets, dim=0)
        return test_sets

    def sample_img_set(self, anomaly_label):
        """
        Samples a new set of images, given the label of the anomaly.
        The sampled images come from a different class than anomaly_label
        """
        # Sample class from 0,...,num_classes-1 while skipping anomaly_label as class
        set_label = np.random.randint(self.num_labels - 1)
        if set_label >= anomaly_label:
            set_label += 1

        # Sample images from the class determined above
        img_indices = np.random.choice(self.img_idx_by_label.shape[1], size=self.set_size, replace=False)
        img_indices = self.img_idx_by_label[set_label, img_indices]
        return img_indices

    def __len__(self):
        return self.img_feats.shape[0]

    def __getitem__(self, idx):
        anomaly = self.img_feats[idx]
        if self.train:  # If train => sample
            img_indices = self.sample_img_set(self.labels[idx])
        else:  # If test => use pre-generated ones
            img_indices = self.test_sets[idx]

        # Concatenate images. The anomaly is always the last image for simplicity
        img_set = torch.cat([self.img_feats[img_indices], anomaly[None]], dim=0)
        indices = torch.cat([img_indices, torch.LongTensor([idx])], dim=0)
        label = img_set.shape[0] - 1

        # We return the indices of the images for visualization purpose. "Label" is the index of the anomaly
        return img_set, indices, label


def visualize_exmp(indices, orig_dataset):
    images = [orig_dataset[idx][0] for idx in indices.reshape(-1)]
    images = torch.stack(images, dim=0)
    images = images * TORCH_DATA_STD + TORCH_DATA_MEANS

    img_grid = torchvision.utils.make_grid(images, nrow=SET_SIZE, normalize=True, pad_value=0.5, padding=16)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(12, 8))
    plt.title("Anomaly examples on CIFAR100")
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()
    plt.close()


class AnomalyPredictor(TransformerPredictor):

    def _calculate_loss(self, batch, mode="train"):
        img_sets, _, labels = batch
        preds = self.forward(img_sets,
                             add_positional_encoding=False)  # No positional encodings as it is a set, not a sequence!
        preds = preds.squeeze(dim=-1)  # Shape: [Batch_size, set_size]
        loss = F.cross_entropy(preds, labels)  # Softmax/CE over set dimension
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


def train_anomaly(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "SetAnomalyTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=100,
                         gradient_clip_val=2)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SetAnomalyTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = AnomalyPredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = AnomalyPredictor(max_iters=trainer.max_epochs * len(train_anom_loader), **kwargs)
        trainer.fit(model, train_anom_loader, val_anom_loader)
        model = AnomalyPredictor.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, train_anom_loader, verbose=False)
    val_result = trainer.test(model, val_anom_loader, verbose=False)
    test_result = trainer.test(model, test_anom_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"],
              "train_acc": train_result[0]["test_acc"]}

    model = model.to(device)
    return model, result


def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row + 1}, Head {column + 1}")
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def visualize_prediction(idx):
    visualize_exmp(indices[idx:idx + 1], test_set)
    print("Prediction:", predictions[idx].item())
    plot_attention_maps(input_data=None, attn_maps=attention_maps, idx=idx)


if __name__ == "__main__":
    DATASET_PATH = "../data"
    # ImageNet statistics
    DATA_MEANS = np.array([0.485, 0.456, 0.406])
    DATA_STD = np.array([0.229, 0.224, 0.225])
    # As torch tensors for later preprocessing
    TORCH_DATA_MEANS = torch.from_numpy(DATA_MEANS).view(1, 3, 1, 1)
    TORCH_DATA_STD = torch.from_numpy(DATA_STD).view(1, 3, 1, 1)

    # Resize to 224x224, and normalize to ImageNet statistic
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(DATA_MEANS, DATA_STD)])
    # Loading the training dataset.
    train_set = CIFAR100(root=DATASET_PATH, train=True, transform=transform, download=True)
    # Loading the test set
    test_set = CIFAR100(root=DATASET_PATH, train=False, transform=transform, download=True)

    os.environ["TORCH_HOME"] = CHECKPOINT_PATH
    get_pretrained_model()
    pretrained_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    # Remove classification layer
    # In some models, it is called "fc", others have "classifier"
    # Setting both to an empty sequential represents an identity map of the final features.
    pretrained_model.fc = nn.Sequential()
    pretrained_model.classifier = nn.Sequential()
    # To GPU
    pretrained_model = pretrained_model.to(device)
    # Only eval, no gradient required
    pretrained_model.eval()
    for p in pretrained_model.parameters():
        p.requires_grad = False

    train_feat_file = os.path.join(CHECKPOINT_PATH, "train_set_features.tar")
    train_set_feats = extract_features(train_set, train_feat_file)
    test_feat_file = os.path.join(CHECKPOINT_PATH, "test_set_features.tar")
    test_feats = extract_features(test_set, test_feat_file)

    print("Train:", train_set_feats.shape)
    print("Test: ", test_feats.shape)

    ## Split train into train+val
    # Get labels from train set
    labels = train_set.targets
    # Get indices of images per class
    labels = torch.LongTensor(labels)
    num_labels = labels.max() + 1
    sorted_indices = torch.argsort(labels).reshape(num_labels, -1)  # [classes, num_imgs per class]
    # Determine number of validation images per class
    num_val_exmps = sorted_indices.shape[1] // 10
    # Get image indices for validation and training
    val_indices = sorted_indices[:, :num_val_exmps].reshape(-1)
    train_indices = sorted_indices[:, num_val_exmps:].reshape(-1)
    # Group corresponding image features and labels
    train_feats, train_labels = train_set_feats[train_indices], labels[train_indices]
    val_feats, val_labels = train_set_feats[val_indices], labels[val_indices]

    SET_SIZE = 10
    test_labels = torch.LongTensor(test_set.targets)
    train_anom_dataset = SetAnomalyDataset(train_feats, train_labels, set_size=SET_SIZE, train=True)
    val_anom_dataset = SetAnomalyDataset(val_feats, val_labels, set_size=SET_SIZE, train=False)
    test_anom_dataset = SetAnomalyDataset(test_feats, test_labels, set_size=SET_SIZE, train=False)

    train_anom_loader = data.DataLoader(train_anom_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4,
                                        pin_memory=True)
    val_anom_loader = data.DataLoader(val_anom_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    test_anom_loader = data.DataLoader(test_anom_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

    _, indices, _ = next(iter(test_anom_loader))
    visualize_exmp(indices[:4], test_set)

    anomaly_model, anomaly_result = train_anomaly(input_dim=train_anom_dataset.img_feats.shape[-1],
                                                  model_dim=256, num_heads=4, num_classes=1, num_layers=4, dropout=0.1,
                                                  input_dropout=0.1, lr=5e-4, warmup=100)

    print(f"Train accuracy: {(100.0 * anomaly_result['train_acc']):4.2f}%")
    print(f"Val accuracy:   {(100.0 * anomaly_result['val_acc']):4.2f}%")
    print(f"Test accuracy:  {(100.0 * anomaly_result['test_acc']):4.2f}%")

    inp_data, indices, labels = next(iter(test_anom_loader))
    inp_data = inp_data.to(device)
    anomaly_model.eval()
    with torch.no_grad():
        preds = anomaly_model.forward(inp_data, add_positional_encoding=False)
        preds = F.softmax(preds.squeeze(dim=-1), dim=-1)
        # Permut input data
        permut = np.random.permutation(inp_data.shape[1])
        perm_inp_data = inp_data[:, permut]
        perm_preds = anomaly_model.forward(perm_inp_data, add_positional_encoding=False)
        perm_preds = F.softmax(perm_preds.squeeze(dim=-1), dim=-1)

    assert (preds[:, permut] - perm_preds).abs().max() < 1e-5, "Predictions are not permutation equivariant"
    print("Preds\n", preds[0, permut].cpu().numpy())
    print("Permuted preds\n", perm_preds[0].cpu().numpy())

    attention_maps = anomaly_model.get_attention_maps(inp_data, add_positional_encoding=False)
    predictions = preds.argmax(dim=-1)

    visualize_prediction(0)

    mistakes = torch.where(predictions != 9)[0].cpu().numpy()
    print("Indices with mistake:", mistakes)
    visualize_prediction(mistakes[-1])
    print("Probabilities:")
    for i, p in enumerate(preds[mistakes[-1]].cpu().numpy()):
        print(f"Image {i}: {100.0 * p:4.2f}%")
