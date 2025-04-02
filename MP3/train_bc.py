import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

def train_model(model, logdir, states, actions, device, discrete):
    model.to(device)
    model.train()

    # Flatten temporal dimensions if needed
    if len(states.shape) > 2:
        states = states.reshape(-1, states.shape[-1])
    if discrete:
        if len(actions.shape) > 1:
            actions = actions.reshape(-1)
    else:
        if len(actions.shape) > 2:
            actions = actions.reshape(-1, actions.shape[-1])

    # Convert to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long if discrete else torch.float32, device=device)

    # Print stats
    print(f"States shape: {states.shape}, Actions shape: {actions.shape}")
    if discrete:
        print(f"Action stats: min={actions.min().item()}, max={actions.max().item()}, mean={actions.float().mean().item():.4f}")
    else:
        print(f"Action stats: min={actions.min().item():.4f}, max={actions.max().item():.4f}, mean={actions.mean().item():.4f}")

    # Warn if actions are nearly constant
    if torch.std(actions.float()) < 1e-4:
        print("⚠️ Warning: Actions are nearly constant — this may lead to poor behavior cloning.")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss() if discrete else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter(str(logdir))

    num_epochs = 100
    batch_size = 64
    num_samples = states.shape[0]

    for epoch in range(num_epochs):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0

        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_states = states[indices]
            batch_actions = actions[indices]

            preds = model(batch_states)

            if discrete:
                batch_actions = batch_actions.squeeze(-1) if batch_actions.ndim > 1 else batch_actions
                loss = criterion(preds, batch_actions)
            else:
                loss = criterion(preds, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (num_samples / batch_size)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"[Epoch {epoch:03d}] Loss: {avg_loss:.6f}")

    writer.close()

