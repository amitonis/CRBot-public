# 🎮 Clash Royale Bot - Quick Start Summary

## ✅ What's Already Done

✔️ Repository cloned successfully  
✔️ All Python dependencies installed:
- PyTorch
- PyAutoGUI
- Roboflow Inference SDK
- NumPy, OpenCV, and all other requirements
- pynput for keyboard control

✔️ Environment file (.env) created  
✔️ Project structure ready  

## 📋 What You Need to Do Next

### 1. Set Up Roboflow (Required)

You need to configure Roboflow for computer vision (detecting troops and cards):

**a) Create Roboflow Account**
- Go to https://roboflow.com/ and sign up

**b) Get Your API Key**
- Go to workspace settings
- Copy your **Private API Key**

**c) Fork Two Workflows**

Click these links and fork them to your workspace:

1. **Troop Detection**: https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiTEx3TjlnOEduenBjWmVYSktKYzEiLCJ3b3Jrc3BhY2VJZCI6Ik5vVUlkM3gyYWRSU0tqaURrM0ZMTzlBSmE1bzEiLCJ1c2VySWQiOiJOb1VJZDN4MmFkUlNLamlEazNGTE85QUphNW8xIiwiaWF0IjoxNzUzODgxNTcyfQ.-ZO7pqc3mBX6W49-uThUSBLdUaCRzM9I8exfEu6-lo8

2. **Card Detection**: https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiMEFmeVpSQ3FSS1dhV1J5QTFGNkciLCJ3b3Jrc3BhY2VJZCI6InJtZHNiY2xlU292aEEwNm15UDFWIiwidXNlcklkIjoiTm9VSWQzeDJhZFJTS2ppRGszRkxPOUFKYTVvMSIsImlhdCI6MTc1Mzg4MjE4Mn0.ceYp4JZoNSIrDkrX2vuc9or3qVakNexseYEgacIrfLA

**d) Update .env File**

Edit the `.env` file in the project root with your credentials:

```bash
ROBOFLOW_API_KEY=your_actual_api_key_here
WORKSPACE_TROOP_DETECTION=your-workspace-name
WORKSPACE_CARD_DETECTION=your-workspace-name
```

### 2. Set Up Docker & Inference Server (Required)

**a) Install Docker Desktop**
- Download from https://www.docker.com/
- Install and make sure it's running

**b) Install Inference CLI**
Open a terminal and run:
```bash
pip install inference-cli
```
(This may take 5-10 minutes)

**c) Start the Inference Server**
```bash
inference server start
```

**d) Verify It's Running**
- Open http://localhost:9001/ in your browser
- You should see the Roboflow Inference page

### 3. Set Up BlueStacks (Required)

**a) Install BlueStacks**
- Download from https://www.bluestacks.com/download.html

**b) Create Pie 64-bit Instance**
- Open BlueStacks
- Click "Multi-instance Manager"
- Create a new "Pie 64-bit" (Android 9) instance

**c) Install Clash Royale**
- Open Google Play Store in BlueStacks
- Install Clash Royale
- Log in to your account

**d) Position the Window Correctly** ⚠️ CRITICAL
- Move BlueStacks to the **right-most side** of your screen
- Make it **stretched/maximized**
- The bot uses hardcoded screen coordinates for Windows!

### 4. Test the Setup (Optional but Recommended)

Test if card detection works:
```bash
python test_cards.py
```

Test if elixir counting works:
```bash
python elixir_verification.py
```

### 5. Run the Bot! 🚀

Once everything is configured:

```bash
python train.py
```

**Important:** Immediately after running the command, click on the BlueStacks window to make it the active window!

To stop training, press **Q** on your keyboard.

## 📁 Important Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script - run this to start the bot |
| `env.py` | Game environment - contains state, actions, rewards |
| `dqn_agent.py` | DQN algorithm - replace this with your algorithm |
| `Actions.py` | Game controls - clicking, screenshots, detection |
| `.env` | Your configuration (API keys, workspace names) |
| `SETUP_GUIDE.md` | Detailed setup instructions |
| `ALGORITHM_GUIDE.md` | Guide for implementing your own algorithm |
| `CHECKLIST.md` | Step-by-step checklist |

## 🎯 Next Steps After Setup

1. **Let the bot play** and observe its behavior
2. **Check the models folder** - checkpoints are saved every 10 episodes
3. **Read ALGORITHM_GUIDE.md** to understand how to implement your custom algorithm
4. **Experiment with the reward function** in `env.py`

## 🔧 Key Components to Customize

When you're ready to implement your algorithm:

### 1. Replace the Learning Algorithm
- Edit `dqn_agent.py` or create a new agent file
- The current DQN can be replaced with PPO, A3C, Genetic Algorithms, etc.

### 2. Modify the Reward Function
- Edit `_compute_reward()` in `env.py`
- Current rewards:
  - -enemy_presence (negative for enemies on field)
  - +elixir_efficiency (reward for good elixir usage)
  - +20 for destroying enemy towers
  - -5 for wasting spells
  - +100 for winning
  - -100 for losing

### 3. Adjust State Representation
- Edit `_get_state()` in `env.py`
- Current state: [elixir, ally_positions, enemy_positions]
- You could add: tower health, time remaining, card cycle info

### 4. Change Action Space
- Edit `get_available_actions()` in `env.py`
- Current: 2,017 discrete actions (4 cards × 18×28 grid + no-op)
- You could: reduce grid size, use continuous actions, hierarchical actions

## 🐛 Common Issues & Solutions

### "ROBOFLOW_API_KEY not set"
→ Edit `.env` file and add your API key

### "No cards detected"
→ Check BlueStacks window position (right-most, stretched)  
→ Verify Inference Server is running (http://localhost:9001/)

### Docker not working
→ Make sure Docker Desktop is running  
→ Restart inference server: `inference server start`

### Bot clicks wrong positions
→ BlueStacks window not positioned correctly  
→ Check window is on right side and stretched

## 📊 Monitoring Training

The bot will print:
```
Episode 1 starting. Epsilon: 1.000
Detected cards: ['Knight', 'Archers', 'Fireball', 'Zap']
Attempting to play Knight
...
Episode 1: Total Reward = 45.23, Epsilon = 0.997
```

- **Epsilon**: Exploration rate (starts at 1.0, decays to 0.01)
- **Total Reward**: Episode performance (higher is better)
- **Models saved**: Every 10 episodes in `models/` folder

## 📚 Additional Resources

- **SETUP_GUIDE.md** - Comprehensive setup instructions with troubleshooting
- **ALGORITHM_GUIDE.md** - How to implement your own RL algorithm
- **CHECKLIST.md** - Step-by-step checklist for setup
- **Original README.md** - Project overview and credits

## ⚠️ Known Issues

From the original developer:
- "Play again" button handling has bugs
- Some minor gameplay issues
- Feel free to contribute fixes!

## 🎓 Learning Resources

If you want to learn more about the algorithms:
- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
- [Reinforcement Learning Book](http://incompleteideas.net/book/the-book-2nd.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

---

## Ready to Start? 

1. ✅ Check that all dependencies are installed (they are!)
2. ⬜ Set up Roboflow (API key + fork workflows)
3. ⬜ Configure .env file
4. ⬜ Start Docker & Inference Server
5. ⬜ Set up BlueStacks & Clash Royale
6. ⬜ Run `python train.py`

Good luck! When you're ready to implement your custom algorithm, let me know what approach you want to take! 🚀
