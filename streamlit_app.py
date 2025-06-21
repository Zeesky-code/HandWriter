import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F


class cVAE(nn.Module):
    def __init__(self, latent_dim=10, num_classes=10):
        super(cVAE, self).__init__()
        self.fc1 = nn.Linear(28*28 + num_classes, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + num_classes, 400)
        self.fc4 = nn.Linear(400, 28*28)
        self.num_classes = num_classes

    def encode(self, x, y):
        x = x.view(-1, 28*28)
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        inputs = torch.cat([x, y_onehot], dim=1)
        h1 = torch.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        inputs = torch.cat([z, y_onehot], dim=1)
        h3 = torch.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


def load_model(path='cvae_mnist.pth', latent_dim=10, num_classes=10):
    model = cVAE(latent_dim, num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()


def generate_images(model, digit, num_images=5, latent_dim=10):
    z = torch.randn(num_images, latent_dim)
    y = torch.full((num_images,), digit, dtype=torch.long)  # digit condition
    with torch.no_grad():
        generated = model.decode(z, y).view(-1, 1, 28, 28)
    return generated


def show_images(images):
    grid = make_grid(images, nrow=5, normalize=True)
    plt.figure(figsize=(10, 2))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    st.pyplot(plt)


st.title("Handwritten Digit Generator 🎨✍️")

digit = st.text_input("Enter a digit (0-9) to generate:", value="0")

if st.button("Generate Images"):
    try:
        digit_int = int(digit)
        if 0 <= digit_int <= 9:
            images = generate_images(model, digit_int)
            show_images(images)
        else:
            st.error("Please enter a valid digit between 0 and 9.")
    except ValueError:
        st.error("Input must be a digit (0-9).")
