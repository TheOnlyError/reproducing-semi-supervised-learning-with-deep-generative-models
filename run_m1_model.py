import torch
import torch.optim as optim
import torch.nn as nn

from load_data import load_data
from models import VAE

from torchvision.utils import save_image

from models.classifier import Classifier

epochs = 50
batch_size = 100
lr = 0.0003
N = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = load_data(batch_size)

# 28x28
features = 784
hidden = 600
latent_features = 50
model = VAE.VAE(features, hidden, latent_features).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.1, 0.001))
criterion = nn.BCELoss(reduction='sum')


def custom_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(model, classifier, dataloader):
    model.train()
    running_loss = 0.0
    for i, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, z, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = custom_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        classifier.train(z.detach().cpu().numpy(), labels.numpy())

    train_loss = running_loss / len(dataloader.dataset)

    # Fitting the classifier
    print('Fitting classifier')
    classifier.fit(N)

    return train_loss


def test(model, classifier, dataloader):
    model.eval()
    running_loss = 0.0
    classifier_loss = 0.0

    with torch.no_grad():
        for i, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, z, mu, logvar = model(data)

            bce_loss = criterion(reconstruction, data)
            loss = custom_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            if i == int(len(test_loader) / dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8],
                                  reconstruction.view(batch_size, 1, 28, 28)[:8]))
                save_image(both.cpu(), f"./data/images/VGAN/MNIST/output{epoch}.png", nrow=num_rows)

            # Validate classifier
            loss = classifier.validate(z.detach().cpu().numpy(), labels.numpy())
            classifier_loss += loss * z.shape[0]

    classifier_loss = classifier_loss / len(dataloader.dataset)

    test_loss = running_loss / len(dataloader.dataset)
    return test_loss, classifier_loss


train_loss = []
test_loss = []
classifierloss = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    classifier = Classifier()
    train_epoch_loss = train(model, classifier, train_loader)
    test_epoch_loss, classifier_epoch_loss = test(model, classifier, test_loader)
    train_loss.append(train_epoch_loss)
    test_loss.append(test_epoch_loss)
    classifierloss.append(classifier_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {test_epoch_loss:.4f}")
    print(f"Classifier accuracy: {classifier_epoch_loss:.4f}")

print(train_loss)
print(test_loss)
print(classifierloss)
