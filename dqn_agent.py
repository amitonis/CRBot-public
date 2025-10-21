import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.action_size = action_size
        self.context_feature_len = 14  # Matches env.extra_state_features
        self.last_context_bucket = None

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done):
        bucket = self._compute_context_bucket(s)
        self.last_context_bucket = bucket
        self.memory.append((s, a, r, s2, done, bucket))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        target_context = self.last_context_bucket
        batch = self._sample_contextual_batch(batch_size, target_context)

        for experience in batch:
            if len(experience) == 6:
                state, action, reward, next_state, done, _ = experience
            else:
                state, action, reward, next_state, done = experience
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state)))
            target_f = self.model(torch.FloatTensor(state))
            target_f = target_f.clone()
            target_f[action] = float(target)

            prediction = self.model(torch.FloatTensor(state))[action]
            loss = self.criterion(prediction, target_f[action].detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filename):
        # Look in models/ directory by default
        path = filename
        if not os.path.isabs(filename):
            path = os.path.join("models", filename)
        try:
            state_dict = torch.load(path, weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)
            self.target_model.load_state_dict(self.model.state_dict())
            self.model.eval()
            print(f"Loaded model weights from {path}")
        except RuntimeError as exc:
            print(f"⚠️ Model load mismatch for {path}: {exc}. Using freshly initialized weights.")

    def _compute_context_bucket(self, state):
        if state is None:
            return None
        slice_len = min(self.context_feature_len, len(state))
        context_slice = state[-slice_len:]
        return tuple(int(round(float(val) * 10)) for val in context_slice)

    def _context_distance(self, bucket_a, bucket_b):
        if bucket_a is None or bucket_b is None:
            return float('inf')
        length = min(len(bucket_a), len(bucket_b))
        return sum(abs(bucket_a[i] - bucket_b[i]) for i in range(length)) / max(length, 1)

    def _sample_contextual_batch(self, batch_size, target_context):
        if target_context is None:
            return random.sample(self.memory, batch_size)

        contextual_experiences = [exp for exp in self.memory if len(exp) == 6 and self._context_distance(exp[5], target_context) <= 8]
        batch = []

        if contextual_experiences:
            select_count = min(len(contextual_experiences), max(1, batch_size // 2))
            batch.extend(random.sample(contextual_experiences, select_count))

        remaining = batch_size - len(batch)
        if remaining > 0:
            selected_ids = {id(exp) for exp in batch}
            remaining_pool = [exp for exp in self.memory if id(exp) not in selected_ids]
            if remaining_pool:
                sample_size = min(len(remaining_pool), remaining)
                batch.extend(random.sample(remaining_pool, sample_size))

        return batch