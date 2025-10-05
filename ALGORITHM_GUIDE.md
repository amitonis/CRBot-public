# 🧠 Implementing Your Own Algorithm

This guide will help you replace the DQN algorithm with your own custom algorithm.

## Current Architecture Overview

The bot currently uses **Deep Q-Learning (DQN)** with the following components:

### 1. State Representation (`env.py`)
The state includes:
- **Elixir count** (0-10)
- **Ally positions** (up to 10 allies, normalized x,y coordinates)
- **Enemy positions** (up to 10 enemies, normalized x,y coordinates)

Total state size: `1 + 2*10 + 2*10 = 41` float values

### 2. Action Space (`env.py`)
Actions are combinations of:
- **Card selection** (0-3, one of 4 cards in hand)
- **X position** (0-17, 18 grid positions)
- **Y position** (0-27, 28 grid positions)
- **No-op action** (do nothing)

Total actions: `4 × 18 × 28 + 1 = 2,017` possible actions

### 3. Reward Function (`env.py`)
Current rewards:
- **-enemy_presence**: Negative reward for enemies on the field
- **+elixir_efficiency**: Reward for effective elixir spending
- **+20**: Destroying an enemy princess tower
- **-5**: Penalty for wasting spells
- **+100**: Victory
- **-100**: Defeat

## 🎯 How to Implement Your Algorithm

### Step 1: Decide on Your Algorithm

Some popular alternatives to DQN:

1. **Policy Gradient Methods**
   - REINFORCE
   - Actor-Critic (A2C, A3C)
   - Proximal Policy Optimization (PPO) ⭐ Recommended
   - Trust Region Policy Optimization (TRPO)

2. **Evolutionary Algorithms**
   - Genetic Algorithms
   - Evolution Strategies
   - NEAT (NeuroEvolution of Augmenting Topologies)

3. **Monte Carlo Methods**
   - Monte Carlo Tree Search (MCTS)
   - Upper Confidence Bounds (UCB)

4. **Other RL Algorithms**
   - SARSA
   - Double DQN
   - Dueling DQN
   - Rainbow DQN

5. **Imitation Learning / Behavior Cloning**
   - Learn from recorded gameplay

### Step 2: Modify `dqn_agent.py`

Replace the DQN implementation with your algorithm. Here's a template:

```python
# your_agent.py
import torch
import torch.nn as nn
import torch.optim as optim

class YourNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(YourNetwork, self).__init__()
        # Define your network architecture
        pass
    
    def forward(self, x):
        # Forward pass
        pass

class YourAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize your model
        self.model = YourNetwork(state_size, action_size)
        
        # Initialize optimizer, memory, hyperparameters, etc.
        pass
    
    def act(self, state):
        """
        Given a state, choose an action
        Returns: action_index (int)
        """
        pass
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience for training
        """
        pass
    
    def train(self, batch_size):
        """
        Train the model using stored experiences
        """
        pass
    
    def save(self, filename):
        """
        Save model weights
        """
        torch.save(self.model.state_dict(), filename)
    
    def load(self, filename):
        """
        Load model weights
        """
        self.model.load_state_dict(torch.load(filename))
```

### Step 3: Modify `train.py`

Update the training script to use your agent:

```python
from your_agent import YourAgent  # Instead of dqn_agent

def train():
    env = ClashRoyaleEnv()
    agent = YourAgent(env.state_size, env.action_size)  # Your agent
    
    # Your training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train(batch_size)
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode}: Reward = {total_reward}")
        
        # Save model periodically
        if episode % 10 == 0:
            agent.save(f"models/model_{episode}.pth")
```

### Step 4: Customize the Reward Function (Optional)

Edit the `_compute_reward()` method in `env.py`:

```python
def _compute_reward(self, state):
    """
    Customize this function to guide your agent's learning
    """
    if state is None:
        return 0
    
    # Example: Your custom reward calculation
    reward = 0
    
    # Reward for elixir management
    elixir = state[0] * 10
    if elixir > 8:
        reward -= 1  # Penalize hoarding elixir
    
    # Reward for board control
    enemy_positions = state[1 + 2 * MAX_ALLIES:]
    enemy_presence = sum(enemy_positions[1::2])
    reward -= enemy_presence * 0.5
    
    # Reward for aggressive play
    ally_positions = state[1:1 + 2 * MAX_ALLIES]
    ally_forward = sum(1 for y in ally_positions[1::2] if y < 0.3)
    reward += ally_forward * 2
    
    return reward
```

### Step 5: Modify State Representation (Optional)

If you want a different state representation, edit `_get_state()` in `env.py`:

```python
def _get_state(self):
    """
    Customize what information your agent sees
    """
    # Current implementation captures:
    # - Screenshot
    # - Elixir count
    # - Detected troops (allies and enemies)
    
    # You could add:
    # - Tower health
    # - Remaining time
    # - Card cycle information
    # - Historical state data
    
    # Example: Add tower health to state
    elixir = self.actions.count_elixir()
    ally_positions = self._get_ally_positions()
    enemy_positions = self._get_enemy_positions()
    tower_health = self._get_tower_health()  # You'd need to implement this
    
    state = np.array([
        elixir / 10.0,
        tower_health['ally_king'] / 100.0,
        tower_health['enemy_king'] / 100.0,
        *ally_positions,
        *enemy_positions
    ], dtype=np.float32)
    
    return state
```

## 📚 Example: Implementing PPO

Here's a skeleton for implementing PPO (a popular algorithm):

```python
# ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Actor head (policy)
        self.actor = nn.Linear(128, action_size)
        
        # Critic head (value function)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_size, action_size):
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        
        self.gamma = 0.99
        self.epsilon_clip = 0.2
        self.K_epochs = 4
        
        self.memory = []
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.model(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        # PPO training logic here
        # Update policy using clipped objective
        # Update value function
        pass
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
    
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
```

## 🔧 Common Modifications

### 1. Action Space Reduction
If 2,017 actions is too many:

```python
# In env.py, reduce grid granularity
self.grid_width = 9   # Instead of 18
self.grid_height = 14  # Instead of 28
# Now only 9 × 14 × 4 + 1 = 505 actions
```

### 2. Continuous Action Space
Use continuous actions instead of discrete:

```python
def act(self, state):
    # Output: [card_prob_0, card_prob_1, card_prob_2, card_prob_3, x, y]
    # x and y are continuous values between 0 and 1
    action = self.model(state)
    card = np.argmax(action[:4])
    x = action[4]
    y = action[5]
    return card, x, y
```

### 3. Hierarchical Actions
Break down decision-making:

```python
# First decide: Should I play a card?
should_play = self.play_model(state)

# If yes, which card?
if should_play:
    card = self.card_model(state)
    
    # Where to place it?
    x, y = self.position_model(state, card)
```

## 📊 Evaluation Metrics

Track these metrics to evaluate your algorithm:

```python
# In train.py
metrics = {
    'total_reward': [],
    'win_rate': [],
    'avg_elixir_efficiency': [],
    'towers_destroyed': [],
    'episode_length': []
}

# Log metrics
if result == "victory":
    metrics['win_rate'].append(1)
else:
    metrics['win_rate'].append(0)

# Calculate rolling average
if len(metrics['win_rate']) >= 100:
    recent_winrate = np.mean(metrics['win_rate'][-100:])
    print(f"Win rate (last 100): {recent_winrate:.2%}")
```

## 🎓 Tips for Success

1. **Start Simple**: Begin with a simpler algorithm and make sure it works
2. **Monitor Learning**: Plot rewards, win rates, and other metrics
3. **Tune Hyperparameters**: Learning rate, discount factor, exploration rate
4. **Curriculum Learning**: Start with easier tasks, gradually increase difficulty
5. **Debug Thoroughly**: Print states, actions, rewards to understand behavior
6. **Save Checkpoints**: Save models frequently to avoid losing progress

## 🔗 Useful Resources

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - Pre-built RL algorithms
- [OpenAI Spinning Up](https://spinningup.openai.com/) - RL tutorial
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Sutton & Barto Book](http://incompleteideas.net/book/the-book-2nd.html) - RL Bible

---

Good luck implementing your algorithm! Feel free to experiment and iterate. 🚀
