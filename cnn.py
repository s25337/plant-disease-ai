class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, **kwargs) #convolution filters
        self.bn = nn.BatchNorm2d(out_chanels)  #batch normalisation

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# The Inception Block allows a neural network to learn multiple types of features (small patterns, large patterns,
# pooled patterns) at the same time by using parallel branches with different convolution types and a pooling layer.
# This is a key idea in the Inception architecture.
#Goal: Extract and preserve the strongest, most important features at different scales.

class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        red_3x3,
        out_3x3,
        red_5x5,
        out_5x5,
        out_pool,
    ):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(                                   #sequential - stack multipule layers
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),  #red_3x3 - reduction before 5x5 convolution
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),    #selects max value from each region
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)

#Goal: Summarize and smooth the spatial information before classification.

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.dropout = nn.Dropout(p=0.7) #During training, this randomly zeroes 70% of the inputs to this layer to prevent overfitting.
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3) #
        self.conv = ConvBlock(in_channels, 128, kernel_size=1) #output 128 channels
        self.fc1 = nn.Linear(2048, 1024) #fully connected leyer (input 2048 flatten vector, output - vector 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1) #Flattens the tensor from shape [batch_size, channels, height, width] to [batch_size, features].
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#Max pool - strongest signal
#Average pool - get all info, average

#extract features and classifies them

class InceptionV1(pl.LightningModule):
  def __init__(self, aux_logits=True, num_classes=1000):
    super(InceptionV1, self).__init__()
    self.aux_logits = aux_logits
    self.conv1 = ConvBlock(
        in_channels=3,
        out_chanels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
    )
    self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
    self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
    self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
    self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
    self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
    self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
    self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
    self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
    self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
    self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
    self.dropout = nn.Dropout(p=0.4)
    self.fc = nn.Linear(1024, num_classes)

    if self.aux_logits:
        self.aux1 = InceptionAux(512, num_classes)
        self.aux2 = InceptionAux(528, num_classes)
    else:
        self.aux1 = self.aux2 = None

    self.train_acc = torchmetrics.Accuracy()
    self.val_acc = torchmetrics.Accuracy()

    self.train_macro_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro')
    self.val_macro_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro')

  def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)

        if self.aux_logits and self.training: # wykorzysujemy auxilary classifiers tylko podczas treningu
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training: # wykorzysujemy auxilary classifiers tylko podczas treningu
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_logits and self.training: # wykorzysujemy auxilary classifiers tylko podczas treningu
            return aux1, aux2, x
        return x

  def configure_optimizers(self):
    optimizer =  optim.SGD(self.parameters(), lr = 0.01)
    return optimizer

  def training_step(self, train_batch, batch_idx):
    inputs, labels = train_batch


    aux1, aux2, outputs = self.forward(inputs.float()) # model podczas treningu zwraca 3 outputy - 2 z auxiliary classifiers i 1 końcowy

    #liczymy funkcje błedu dla wszystkich 3 outputów
    aux1_loss = self.loss_function(aux1, labels)
    aux2_loss = self.loss_function(aux2, labels)
    output_loss = self.loss_function(outputs, labels)

    #ostateczna funkcja błedu jest sumą funkcji błędu dla wszyskich outputów
    loss = aux1_loss + aux2_loss + output_loss

    self.log('train_loss', loss, on_step= True, on_epoch = True)

    outputs = F.softmax(outputs, dim =1)

    self.train_acc(outputs, labels)
    self.log('train_acc', self.train_acc, on_epoch=True, on_step= False)

    self.train_macro_f1(outputs, labels)
    self.log('train_macro_f1', self.train_macro_f1, on_epoch=True, on_step= False)


    return loss

  def validation_step(self, val_batch, batch_idx):
    inputs, labels = val_batch


    outputs = self.forward(inputs.float())
    loss = self.loss_function(outputs, labels)

    self.log('val_loss', loss,  on_step= True, on_epoch = True)


    outputs = F.softmax(outputs, dim =1)

    self.val_acc(outputs, labels)
    self.log('val_acc', self.val_acc, on_epoch=True, on_step= False)

    self.val_macro_f1(outputs, labels)
    self.log('val_macro_f1', self.val_macro_f1, on_epoch=True, on_step= False)

    return loss

#This is a Residual Block (ResBlock), a core building block of ResNet architectures, designed to help train very deep neural networks by enabling easier gradient flow via skip connections.

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut