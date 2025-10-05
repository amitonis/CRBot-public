# ✅ Setup Progress Checklist

Use this checklist to track your setup progress.

## Initial Setup

- [x] Repository cloned
- [x] Python dependencies installed
- [x] `.env` file created
- [ ] Roboflow account created
- [ ] Roboflow API key obtained
- [ ] Troop Detection workflow forked
- [ ] Card Detection workflow forked
- [ ] `.env` file configured with API key and workspace names
- [ ] Docker Desktop installed
- [ ] Docker Desktop running
- [ ] Inference CLI installed
- [ ] Inference Server started and verified (http://localhost:9001/)
- [ ] BlueStacks installed
- [ ] BlueStacks Pie 64-bit instance created
- [ ] Clash Royale installed in BlueStacks
- [ ] BlueStacks window positioned correctly (right-most, stretched)

## Testing

- [ ] Tested card detection with `python test_cards.py`
- [ ] Elixir counting works correctly
- [ ] Bot can click and play cards

## Ready to Run!

- [ ] All setup steps completed
- [ ] Ready to run `python train.py`

---

## Important Notes

### Roboflow Workflows to Fork:

1. **Troop Detection**: https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiTEx3TjlnOEduenBjWmVYSktKYzEiLCJ3b3Jrc3BhY2VJZCI6Ik5vVUlkM3gyYWRSU0tqaURrM0ZMTzlBSmE1bzEiLCJ1c2VySWQiOiJOb1VJZDN4MmFkUlNLamlEazNGTE85QUphNW8xIiwiaWF0IjoxNzUzODgxNTcyfQ.-ZO7pqc3mBX6W49-uThUSBLdUaCRzM9I8exfEu6-lo8

2. **Card Detection**: https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiMEFmeVpSQ3FSS1dhV1J5QTFGNkciLCJ3b3Jrc3BhY2VJZCI6InJtZHNiY2xlU292aEEwNm15UDFWIiwidXNlcklkIjoiTm9VSWQzeDJhZFJTS2ppRGszRkxPOUFKYTVvMSIsImlhdCI6MTc1Mzg4MjE4Mn0.ceYp4JZoNSIrDkrX2vuc9or3qVakNexseYEgacIrfLA

### Your `.env` file should look like:

```bash
ROBOFLOW_API_KEY=your_actual_api_key_from_roboflow
WORKSPACE_TROOP_DETECTION=your-workspace-name
WORKSPACE_CARD_DETECTION=your-workspace-name
```

### Docker Commands:

```bash
# Install inference CLI
pip install inference-cli

# Start inference server
inference server start

# Verify server is running
# Open: http://localhost:9001/
```

### Running the Bot:

```bash
# Test card detection (optional)
python test_cards.py

# Start training
python train.py

# Stop training: Press 'Q' on keyboard
```

---

## Next Steps After Setup

Once everything is working, you can:

1. **Watch the bot play** and learn from its mistakes
2. **Monitor the training** - models are saved every 10 episodes in `models/`
3. **Implement your custom algorithm** - modify `dqn_agent.py` and `env.py`
4. **Adjust the reward function** - change how the bot learns in `env.py`
5. **Fine-tune hyperparameters** - epsilon decay, learning rate, etc.

Good luck! 🎮🤖
