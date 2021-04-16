import torch
import torch.optim as optim
import torch.nn as nn

from load_data import load_data
from models import VAEM2
import numpy as np
from torchvision.utils import save_image

from models.classifier import Classifier

epochs = 50
batch_size = 100
lr = 0.0003
N = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader = load_data(batch_size)

features = 784
hidden = 500
latent_features = 50
model = VAEM2.VAE(features, hidden, latent_features).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.1, 0.001))
criterion = nn.BCELoss(reduction='sum')
second_criterion = nn.CrossEntropyLoss(reduction='sum')


def custom_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    label_correct = 0.0

    y_labels = torch.tensor([[i for i in range(10)] for _ in range(100)]).cuda()

    for i, data in enumerate(dataloader):
        data, labels = data
        data = data.to(device)
        labels = labels.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, z, mu, logvar, pi = model(data)

        bce_loss = criterion(reconstruction, data)

        with torch.no_grad():
            label_correct += (torch.argmax(pi, 1) == labels).sum()

        if i < N:
            loss = custom_loss(bce_loss, mu, logvar)
            loss += second_criterion(pi, labels)
        else:
            loss = custom_loss(bce_loss, mu, logvar)
            U = 0.0

            for y in range(10):
                y_labels = torch.tensor([y for _ in range(pi.shape[0])]).cuda()
                L = loss + second_criterion(pi, y_labels)

                q_y_x = pi[:, y].sum()
                # H = torch.heaviside(pi[:, y], torch.tensor([0.0], requires_grad=False).cuda())

                U = U + q_y_x + L #+ H.sum()

            loss = U

        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(dataloader.dataset)
    classifier_loss = label_correct / len(dataloader.dataset)

    return train_loss, classifier_loss


def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    label_correct = 0.0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data, labels = data
            data = data.to(device)
            labels = labels.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, z, mu, logvar, pi = model(data)

            bce_loss = criterion(reconstruction, data)

            label_correct += (torch.argmax(pi, 1) == labels).sum()

            second_loss = second_criterion(pi, labels)
            loss = custom_loss(bce_loss, mu, logvar) + second_loss
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_loader) / dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8],
                                  reconstruction.view(batch_size, 1, 28, 28)[:8]))
                save_image(both.cpu(), f"./data/images/VGAN/MNIST/output{epoch}.png", nrow=num_rows)

    classifier_loss = label_correct / len(dataloader.dataset)

    test_loss = running_loss / len(dataloader.dataset)

    return test_loss, classifier_loss


train_loss = []
test_loss = []
test_classifierloss = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss, train_classifier_loss = fit(model, train_loader)
    val_epoch_loss, classifier_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    test_loss.append(val_epoch_loss)
    test_classifierloss.append(classifier_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")
    print(f"Train Classifier accuracy: {train_classifier_loss:.4f}")
    print(f"Classifier accuracy: {classifier_epoch_loss:.4f}")

print(train_loss)
print(test_loss)
print(test_classifierloss)
