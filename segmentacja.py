mask = transforms.Compose([
    # other transforms
    transforms.ToTensor(),
    lambda x: x*255
])(mask).squeeze()

#Converts a PIL image or a NumPy array into a PyTorch tensor.

class SegmentationDataset(Dataset):

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            preprocessing=None,
    ):
        self.ids = self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = list(range(len(self.CLASSES)))

        self.preprocessing = preprocessing
        self.image_transform = transforms.Compose([
                                    transforms.CenterCrop((352,480)),
                                    transforms.ToTensor()])
        self.mask_transform = transforms.Compose([
                                    transforms.CenterCrop((352,480)),
                                    transforms.ToTensor(),
                                    lambda x: x*255])

    def __getitem__(self, i):

        # read data
        image = Image.open(self.images_fps[i])
        mask =  Image.open(self.masks_fps[i])

        # extract certain classes from mask (e.g. cars)
        image = self.image_transform(image)
        mask = self.mask_transform(mask).squeeze()


        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

class DataModule(pl.LightningDataModule):
    def __init__(self, x_train_dir, y_train_dir,x_val_dir, y_val_dir, x_test_dir, y_test_dir, batch_size = 16):
        super().__init__()
        self.x_train_dir = x_train_dir
        self.y_train_dir = y_train_dir

        self.x_val_dir = x_val_dir
        self.y_val_dir = y_val_dir

        self.x_test_dir = x_test_dir
        self.y_test_dir = y_test_dir

        self.batch_size = batch_size

    def setup(self, stage = None):

        self.train_dataset = SegmentationDataset(self.x_train_dir, self.y_train_dir)
        self.val_dataset = SegmentationDataset(self.x_val_dir, self.y_val_dir)
        self.test_dataset = SegmentationDataset(self.x_test_dir, self.y_test_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class SegmentationModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes):
        super().__init__()


        self.model = model = smp.Unet(encoder_name='resnet34',encoder_weights='imagenet', classes=out_classes)
# TODO: utwórz model do segmentacji korzystając z biblioteki smp

        self.loss_fn = DiceLoss()
        self.out_classes = out_classes

    def forward(self, image):
        mask = self.model(image)
        return mask

    def training_step(self, batch, batch_idx):

        image, mask = batch

        pred_mask = self.forward(image)

        loss = self.loss_fn(pred_mask.float(), mask.to(torch.int64))

        pred_mask = torch.argmax(pred_mask,dim=1)

        iou_score = smp_score(pred_mask, mask, reduction="mean")


        self.log('train_loss', loss, on_step= True, on_epoch = True)
        self.log('iou_score', iou_score, on_step= True, on_epoch = True)


        return loss

    def validation_step(self, batch, batch_idx):

        image, mask = batch

        pred_mask = self.forward(image)

        loss = self.loss_fn(pred_mask.float(), mask.to(torch.int64))



        self.log('val_loss', loss, on_step= True, on_epoch = True)

        return loss



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


#Segmentacja z augmentacja
class SegmentationDatasetWithAugmentation(Dataset):

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = list(range(len(self.CLASSES)))

        self.preprocessing = preprocessing
        # TODO: dodaj augmentację
        self.image_transform = transforms.Compose([
                                    transforms.CenterCrop((352,480)),
                                    transforms.RandomAutocontrast(0.5),
                                    transforms.RandomRotation(45),
                                    transforms.GaussianBlur((7,7)),
                                    transforms.RandomVerticalFlip(0.5),
                                    transforms.ToTensor()])
        self.image_transform = transforms.Compose([
                                    transforms.CenterCrop((352,480)),
                                    transforms.ToTensor()])
        self.mask_transform = transforms.Compose([
                                    transforms.CenterCrop((352,480)),
                                    transforms.ToTensor(),
                                    lambda x: x*255])

    def __getitem__(self, i):

        # read data
        image = Image.open(self.images_fps[i])
        mask = Image.open(self.masks_fps[i])

        # extract certain classes from mask (e.g. cars)
        image = self.image_transform(image)
        mask = self.mask_transform(mask).squeeze()



        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

class DataModuleWithAugmentation(pl.LightningDataModule):
    def __init__(self, x_train_dir, y_train_dir,x_val_dir, y_val_dir, x_test_dir, y_test_dir, batch_size = 16):
        super().__init__()
        self.x_train_dir = x_train_dir
        self.y_train_dir = y_train_dir

        self.x_val_dir = x_val_dir
        self.y_val_dir = y_val_dir

        self.x_test_dir = x_test_dir
        self.y_test_dir = y_test_dir

        self.batch_size = batch_size

    def setup(self, stage = None):

        self.train_dataset = SegmentationDatasetWithAugmentation(self.x_train_dir, self.y_train_dir)
        self.val_dataset = SegmentationDatasetWithAugmentation(self.x_val_dir, self.y_val_dir)
        self.test_dataset = SegmentationDatasetWithAugmentation(self.x_test_dir, self.y_test_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)