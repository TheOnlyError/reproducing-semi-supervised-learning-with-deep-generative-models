import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, features, hidden, latent_features):
        super(VAE, self).__init__()

        # encoder
        self.enc1 = nn.Sequential(
            nn.Linear(in_features=features, out_features=hidden, bias=True),
            nn.Softplus(),
            nn.Linear(in_features=hidden, out_features=hidden, bias=True),
            nn.Softplus()
        )
        self.enc_mean = nn.Sequential(
            nn.Linear(in_features=hidden, out_features=latent_features, bias=True),
        )

        self.enc_log_var = nn.Sequential(
            nn.Linear(in_features=hidden, out_features=latent_features, bias=True),
        )

        # decoder
        self.dec1 = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=hidden, bias=True),
            nn.Softplus(),
            nn.Linear(in_features=hidden, out_features=hidden, bias=True),
            nn.Softplus(),
        )
        self.dec2 = nn.Sequential(
            nn.Linear(in_features=hidden, out_features=features, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # encoding
        x = self.enc1(x)
        mu = self.enc_mean(x)
        log_var = self.enc_log_var(x)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + (eps * std)

        # decoding
        x = self.dec1(z)
        reconstruction = self.dec2(x)
        return reconstruction, z, mu, log_var