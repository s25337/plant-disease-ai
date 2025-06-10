import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

#Generator - sieć która z szumu generuje obrazy, w naszym przypadku zwykła sieć fully-connected ale możnaby użyć sieci splotowych z upsamlingiem go obrazów.

#Przykładowe implementacje generatorów w torchu : https://github.com/eriklindernoren/PyTorch-GAN Warto zwrócić uwagę na np. na DCGAN - prosty GAN dla obrazów

#Uwaga: w generatorach warstwy BatchNorm sią całkiem istotne

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

g = Generator(100,(1,28,28))

#Dyskryninator - zwykła sieć klasyfikacyjna, w naszym wypadku też FC ale można użyć dowonlych sieci do klasyfikacji obrazu np. z bibliteki TIMM. Przyjmuje obraz jako wejście, zwraca prawdopoobieństwo że zdjęcie jest prawdziwe (od 0 do 1)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class GAN(LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    # Funkcja błędu zawsze taka sama
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # samplujemy szum
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # trenujemy generator
        if optimizer_idx == 0:



            # TODO: wygeneruj fałszywe obrazy za pomocą dyskryminatora z szumu(z)
            generated_imgs = self(z)

            # TODO: policz predykcje dyskryminatora dla wygenerowanych obrazów
            pred = self.discriminator(generated_imgs)


            # TODO: stwórz macierz ZER (wszystkie wygenerowane obrazy powinniśmy rozpoznać jako  fałszywe więc mają etykietę 1)
            #pierwszy wymiar macierzy powinien odpowiadać ilości obrazów w batchu, drugi wymiar powinien być ustawiony na 1
            true = torch.ones(imgs.size(0), 1)
            true = true.type_as(imgs)


            # TODO: policz funkcje błędu pomiedzy wartościami przewidzianymi przez dyskryminator (preds) a wartościami faktycznym (true)

            g_loss =  self.adversarial_loss(pred, true)

            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # trenujemy dyskryminator
        if optimizer_idx == 1:

            # ---Liczymy jak dobrze model jest w stanie rozpoznać obrazy prawdziwe--
            # TODO: policz predykcje dyskryminatora dla prawdziwych obrazów (imgs)
            pred = self.discriminator(imgs)

            # TODO: tworzymy macierz JEDYNEK - wszyskie obrazy powinny być rozpoznane jako prawdziwe
            # pierwszy wymiar macierzy powinien odpowiadać ilości obrazów w batchu, drugi wymiar powinien być ustawiony na 1
            true = torch.ones(imgs.size(0), 1).type_as(imgs)

            true = true.type_as(imgs)

            # TODO: policz funkcje błędu pomiedzy wartościami przewidzianymi przez dyskryminator (preds) a wartościami faktycznym (true)
            real_loss = self.adversarial_loss(pred, true)

            # ---Liczymy jak dobrze model jest w stanie rozpoznać obrazy fałszywe---?
            #TODO: wygeneruj fałszywe obrazy za pomocą generatora z szumu (z)
            generated_imgs = self(z)

            #TODO: policz predykcje dyskryminatora dla wygenerowanych obrazów
            pred = self.discriminator(generated_imgs)
            # TODO: tworzymy macierz ZER - wszyskie obrazy wygenerowane przez dyskryminatora powinny być rozpoznane jako fałszywe
            true  = torch.zeros(imgs.size(0), 1).type_as(imgs)
            true = true.type_as(imgs)

            # TODO: policz funkcje błędu pomiedzy wartościami przewidzianymi przez dyskryminator (preds) a wartościami faktycznym (true)

            fake_loss = self.adversarial_loss(pred, true)

            # finalna funkcja błędy dyskryminatora powinna być średnią dwóch funkcji błędu
            d_loss  = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        #tworzymy dwa optymalizatory - jeden dla generatora jeden dla dyskryminatora
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # wyświetlamy przykładowe obrazy na tensorboardzie
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)