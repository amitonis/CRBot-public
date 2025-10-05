# Clash Royale Bot - Complete Setup Guide

This guide will help you set up the Clash Royale bot on your Windows machine.

## 📋 Prerequisites Checklist

Before starting, make sure you have:

- ✅ Windows OS
- ✅ Python 3.12 installed ([Download](https://www.python.org/downloads/windows/))
- ✅ Docker Desktop installed ([Download](https://www.docker.com/))
- ✅ BlueStacks installed ([Download](https://www.bluestacks.com/download.html))
- ✅ VS Code (or your preferred code editor)
- ✅ A Roboflow account ([Sign up](https://www.roboflow.com/))

## 🚀 Step-by-Step Setup

### 1. Install Python Dependencies

Open a terminal in the project directory and run:

```powershell
# Install PyTorch (CPU version for Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r requirements.txt

# Install pynput for keyboard control
pip install pynput
```

### 2. Set Up Roboflow

#### A. Get Your Roboflow API Key

1. Go to [Roboflow](https://roboflow.com/) and sign up/login
2. Navigate to your workspace settings
3. Find and copy your **Private API Key**

#### B. Fork the Detection Workflows

You need to fork two workflows:

1. **Troop Detection Workflow**:
   - Click [here](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiTEx3TjlnOEduenBjWmVYSktKYzEiLCJ3b3Jrc3BhY2VJZCI6Ik5vVUlkM3gyYWRSU0tqaURrM0ZMTzlBSmE1bzEiLCJ1c2VySWQiOiJOb1VJZDN4MmFkUlNLamlEazNGTE85QUphNW8xIiwiaWF0IjoxNzUzODgxNTcyfQ.-ZO7pqc3mBX6W49-uThUSBLdUaCRzM9I8exfEu6-lo8)
   - Click "Fork" to add it to your workspace
   - Note your workspace name (appears in the URL or workflow settings)

2. **Card Detection Workflow**:
   - Click [here](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiMEFmeVpSQ3FSS1dhV1J5QTFGNkciLCJ3b3Jrc3BhY2VJZCI6InJtZHNiY2xlU292aEEwNm15UDFWIiwidXNlcklkIjoiTm9VSWQzeDJhZFJTS2ppRGszRkxPOUFKYTVvMSIsImlhdCI6MTc1Mzg4MjE4Mn0.ceYp4JZoNSIrDkrX2vuc9or3qVakNexseYEgacIrfLA)
   - Click "Fork" to add it to your workspace
   - Note your workspace name

#### C. Create .env File

1. Copy the example environment file:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Edit `.env` and add your credentials:
   ```bash
   ROBOFLOW_API_KEY=your_actual_api_key_here
   WORKSPACE_TROOP_DETECTION=your-workspace-name
   WORKSPACE_CARD_DETECTION=your-workspace-name
   ```

### 3. Set Up Docker & Roboflow Inference Server

1. **Start Docker Desktop**
   - Make sure Docker Desktop is running

2. **Install Inference CLI**
   
   Open Docker Desktop's terminal (or any terminal) and run:
   ```bash
   pip install inference-cli
   ```
   ⚠️ This may take several minutes - don't worry if it seems stuck!

3. **Start the Inference Server**
   ```bash
   inference server start
   ```

4. **Verify the Server**
   - Open your browser and go to: http://localhost:9001/
   - You should see the Roboflow Inference page

### 4. Set Up BlueStacks Emulator

1. **Open BlueStacks**

2. **Create a New Instance**
   - Click the "Multi-instance Manager" (3rd icon or in the 3 dots menu)
   - Click "+ Instance"
   - Select **"Fresh Instance"**
   - Choose **"Pie 64-bit"** (Android 9)
   - Create and start the instance

3. **Install Clash Royale**
   - Open Google Play Store in BlueStacks
   - Search for "Clash Royale"
   - Install and open it

4. **Configure BlueStacks Window** (IMPORTANT!)
   - Resize and position the BlueStacks window to the **right-most side of your screen**
   - Make it **stretched** (see the GIF in the main README)
   - The bot relies on specific screen coordinates, so positioning is crucial!

5. **Optional: Disable Ads**
   - Click Settings (gear icon)
   - Go to Preferences
   - Disable "Allow BlueStacks to show Ads during gameplay"

### 5. Configure Clash Royale

1. Log in to your Clash Royale account (or create a new one)
2. Navigate to the main menu
3. Click on "Battle" to be ready to start

## 🎮 Running the Bot

### Test Card Detection (Optional)

Before running the full bot, you can test if card detection works:

```powershell
python test_cards.py
```

### Start Training

1. **Make sure BlueStacks with Clash Royale is running and positioned correctly**
2. **Run the training script:**
   ```powershell
   python train.py
   ```
3. **Immediately make BlueStacks the front-most window** (click on it)
4. The bot will start playing!

### Stop the Bot

- Press `Q` on your keyboard to gracefully stop the training

## 📁 Project Structure

```
Clash-Royale-Bot/
├── Actions.py              # Game actions (clicks, screenshots, detection)
├── env.py                  # Clash Royale environment for RL
├── dqn_agent.py           # Deep Q-Network agent
├── train.py               # Main training script
├── test_cards.py          # Test card detection
├── elixir_verification.py # Test elixir counting
├── main_images/           # Reference images for detection
├── screenshots/           # Runtime screenshots
├── models/                # Saved model checkpoints
├── requirements.txt       # Python dependencies
└── .env                   # Your configuration (DO NOT COMMIT!)
```

## 🛠️ How It Works

1. **Computer Vision**: Uses Roboflow workflows to detect:
   - Troops on the battlefield (allies and enemies)
   - Cards in hand
   - Game state (victory/defeat)
   - Elixir count

2. **Reinforcement Learning**: Uses Deep Q-Learning (DQN) to:
   - Learn which cards to play
   - Learn where to place troops
   - Optimize strategy over time

3. **Automation**: Uses PyAutoGUI to:
   - Click cards and battlefield positions
   - Navigate menus
   - Handle game flow

## 🐛 Troubleshooting

### "No cards detected" or "All cards Unknown"
- Make sure BlueStacks is positioned correctly (right-most, stretched)
- Check that the Roboflow Inference Server is running (http://localhost:9001/)
- Verify your `.env` file has the correct workspace names

### "ROBOFLOW_API_KEY environment variable is not set"
- Make sure you copied `.env.example` to `.env`
- Check that your API key is correctly set in the `.env` file

### Docker/Inference Server issues
- Make sure Docker Desktop is running
- Restart the inference server: `inference server start`
- Check Docker logs for any errors

### Bot clicks wrong positions
- This usually means BlueStacks window is not positioned correctly
- The coordinates are hardcoded for Windows at specific positions
- Make sure the window is on the right-most side and stretched

## 📝 Known Issues

As noted in the README:
- The bot has issues with the "play again" functionality
- Some minor gameplay bugs exist
- Feel free to contribute fixes!

## 🎯 Next Steps

Once you have the bot running, you can:
1. Let it train for multiple episodes
2. Watch the epsilon value decrease (exploration vs exploitation)
3. Models are saved every 10 episodes in the `models/` folder
4. Modify the reward function in `env.py` to change behavior
5. Implement your own algorithm (as you mentioned!)

## 📚 Additional Resources

- [Roboflow Documentation](https://docs.roboflow.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [DQN Paper](https://arxiv.org/abs/1312.5602)

---

**Ready to implement your own algorithm?** The key files to modify are:
- `dqn_agent.py` - Replace with your algorithm
- `env.py` - Modify the reward function and state representation
- `train.py` - Adjust training loop as needed

Good luck! 🚀
