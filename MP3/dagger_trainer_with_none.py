import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from framestack import FrameStack
from evaluation import val

def dagger_trainer(env, learner, query_expert, device, num_iters=30, episodes_per_iter=10, max_steps=200):
    dataset = []
    rewards = []
    max_dataset_size = 2000  # Maximum number of examples to keep

    optimizer = optim.Adam(learner.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    # Removed both action persistence and penalty

    for iter_num in range(num_iters):
        print(f"\n=== DAgger Iteration {iter_num + 1}/{num_iters} ===")

        # Collect new data
        new_data = []  # Temporary list for new examples
        for ep in range(episodes_per_iter):
            obs = env.reset()
            if isinstance(obs, tuple):  # gymnasium
                obs = obs[0]

            for t in range(max_steps):
                # Get learner's action (no persistence)
                img = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Always select new action
                with torch.no_grad():
                    logits = learner(img)
                    probs = torch.softmax(logits, dim=1)
                    learner_action = torch.argmax(probs, dim=1).item()

                # Get expert's action
                state = torch.tensor(env.unwrapped.state).float().to(device)
                expert_action = query_expert(state)

                # Store expert's action in new_data
                new_data.append((img.squeeze(0), torch.tensor(expert_action).long()))

                # Step with learner's action
                obs, reward, done, truncated, info = env.step(learner_action)
                if done or truncated:
                    break

        # Combine old and new data, then randomly sample if too large
        dataset.extend(new_data)
        if len(dataset) > max_dataset_size:
            dataset = random.sample(dataset, max_dataset_size)

        print(f"Dataset size: {len(dataset)} examples (max size: {max_dataset_size})")

        # Train learner on aggregated dataset
        learner.train()
        if not dataset:
             print("Dataset is empty, skipping training.")
             continue

        random.shuffle(dataset)
        batch_size = 64

        for epoch in range(4):
            total_loss = 0
            num_batches = 0
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                if not batch:
                    continue
                obs_batch = torch.stack([item[0] for item in batch]).to(device)
                act_batch = torch.tensor([item[1] for item in batch]).to(device)

                optimizer.zero_grad()
                pred = learner(obs_batch)
                loss = criterion(pred, act_batch)
                # No action change penalty

                loss.backward()
                torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"[Iter {iter_num+1}] Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            else:
                print(f"[Iter {iter_num+1}] Epoch {epoch+1}: No batches processed.")

        # Evaluate with updated max_steps
        print("Evaluating...")
        eval_envs = [FrameStack(gym.make('VisualCartPole-v2'), 4) for _ in range(1)]
        print("Evaluating Successfully...")
        avg_reward = val(learner, device, eval_envs, max_steps, visual=False)
        rewards.append(avg_reward)

    # Plot reward
    plt.figure(figsize=(7,5))
    plt.plot(rewards, marker='o')
    plt.xlabel("Number of Expert-Labelled Environment Steps")
    plt.ylabel("Average Reward (5 episodes)")
    plt.title("DAgger Policy Learning Progress (Baseline)")
    plt.grid(True)
    plt.savefig("dagger_progress_baseline.png")
    print("✅ Saved reward plot to dagger_progress_baseline.png")