import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from skimage import io, color
import lightning.pytorch as pl

BATCH_SIZE = 32


class ImageColorizerDataset(Dataset):
    def __init__(
        self,
        root_dir,
        train=True,
        n_samples=1281167,
        transform=None,
        target_transform=None,
    ):
        self.root_dir = root_dir
        self.n_samples = n_samples
        self.train = train
        self.annotations_file = pd.read_csv(
            os.path.join(root_dir, "train.csv" if train else "test.csv"),
            header=None,
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        """Loads an image from a file, converts to LAB color space, and returns (L, [A; B])."""
        image_path = os.path.join(self.root_dir, self.annotations_file.iloc[idx, 0])

        rgb_img = io.imread(image_path)
        lab_img = color.rgb2lab(rgb_img)

        # split into L, A, B channels
        l, ab = lab_img[:, :, 0], lab_img[:, :, 1:]

        if self.transform:
            l = self.transform(l)
        if self.target_transform:
            ab = self.target_transform(ab)

        return l.float(), ab.float()


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_data = ImageColorizerDataset(
    root_dir="./data",
    train=True,
    transform=transform,
    target_transform=transform,
    n_samples=1281167,
)
val_data = ImageColorizerDataset(
    root_dir="./data",
    train=False,
    transform=transform,
    target_transform=transform,
    n_samples=50000,
)

train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
)
val_dataloader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)


class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
        act_fn: callable = nn.GELU,
    ):
        super().__init__()

        c_hid = base_channel_size

        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 -> 32x32
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 -> 16x16
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 -> 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 -> 4x4
            act_fn(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_output_channels: int,
        base_channel_size: int,
        act_fn: callable = nn.GELU,
    ):
        super().__init__()

        c_hid = base_channel_size

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # 4x4 -> 8x8
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # 8x8 -> 16x16
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # 16x16 -> 32x32
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                2 * c_hid,
                c_hid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # 32x32 -> 64x64
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.net(x)

        return x


class LitAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        base_channel_size: int = 32,
        num_input_channels: int = 1,
        num_output_channels: int = 2,
        act_fn: callable = nn.GELU,
        width: int = 64,
        height: int = 64,
    ):
        super().__init__()
        self.encoder = encoder_class(
            num_input_channels=num_input_channels,
            base_channel_size=base_channel_size,
            act_fn=act_fn,
        )
        self.decoder = decoder_class(
            num_output_channels=num_output_channels,
            base_channel_size=base_channel_size,
            act_fn=act_fn,
        )

    def _get_loss(self, batch):
        l, ab = batch
        ab_hat = self.encoder(l)
        ab_hat = self.decoder(ab_hat)
        loss = F.mse_loss(ab_hat, ab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    autoencoder = LitAutoEncoder(encoder_class=Encoder, decoder_class=Decoder)
    trainer = pl.Trainer(max_epochs=128, accelerator="gpu", num_nodes=1, devices=4, strategy="ddp")
    trainer.fit(
        autoencoder,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
