import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ConvNet

# Détection du device (GPU si disponible sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fonction d'entraînement
def train(model, train_loader, perm=torch.arange(784).long(), n_epoch=1):
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters())
    
    for epoch in range(1, n_epoch + 1):
        for i, (data, target) in enumerate(train_loader, 1):
            # Envoi sur le device
            data, target = data.to(device), target.to(device)

            # Permutation circulaire de Toeplitz
            data = data.view(-1, 28*28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)

            # Descente de gradient
            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            # Affichage tous les 100 steps
            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

# Fonction de test
def test(model, test_loader, perm=torch.arange(784).long()):
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Envoi sur le device
            data, target = data.to(device), target.to(device)

            # Permutation circulaire de Toeplitz
            data = data.view(-1, 28*28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)

            # Prédiction
            logits = model(data)

            # Accumulation du loss (somme)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()

            # Calcul des bonnes prédictions
            pred = torch.argmax(logits, dim=1)
            correct += pred.eq(target).sum().item()

    # Moyenne du loss
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy*100:.2f}%)")
    return test_loss, accuracy

# Exemple de point d'entrée
def main():
    # Transformations MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Chargement des données
    train_loader = DataLoader(
        datasets.MNIST("../../data/raw", train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST("../../data/raw", train=False, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    # Initialisation du modèle
    model = ConvNet(input_size=(1,28,28), n_kernels=6, output_size=10)

    # Permutation de base
    perm = torch.arange(784).long()

    # Entraînement et test
    train(model, train_loader, perm, n_epoch=10)
    test(model, test_loader, perm)

    # Sauvegarde du modèle entraîné
    torch.save(model.state_dict(), "../../model/convnet_mnist.pt")

if __name__ == "__main__":
    main()
