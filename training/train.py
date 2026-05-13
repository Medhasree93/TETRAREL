import torch
import torch.nn as nn
import torch.optim as optim



def train_model(model, train_loader, epochs=100, lr=0.002):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()

            output, _ = model(x)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
