import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Détection du device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
print(f"Using device: {device}")

# Définition des transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Chargement des datasets
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data/raw", download=True, train=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data/raw", download=True, train=False, transform=transform),
    batch_size=64, shuffle=True
)

# Affichage des 5 premières images
batch = next(iter(train_loader))
images = batch[0][:5]  # 5 premières images
labels = batch[1][:5]  # 5 premiers labels

# Création d'une figure avec 5 sous-graphiques
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    # Afficher l'image (enlever la dimension du canal avec squeeze())
    axes[i].imshow(images[i].squeeze(), cmap='gray')
    axes[i].set_title(f'Label: {labels[i]}')
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png')  # Sauvegarde la figure dans un fichier
plt.close()  # Ferme la figure pour libérer la mémoire
print("Images sauvegardées dans mnist_samples.png")
