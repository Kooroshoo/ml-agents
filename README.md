# WalkerAgent Documentation

#### Initial Scenario:

![Initial](https://github.com/user-attachments/assets/cb34ef31-ee08-4f09-8b13-a63e024fa21f)

#### Train to Stand: 

![standing](https://github.com/user-attachments/assets/c5ef113d-3c6a-4cd6-ae77-e64e1b1e0a4f)

#### Train to Walk Forward:

## Training the Environment

### Observation Space

#### Body Part Observations
- **Position:** `16 body parts × 3 coordinates = 48 values`
- **Rotation:** `16 body parts × 4 values = 64 values`
- **Velocity:** `16 body parts × 3 values = 48 values`
- **Angular Velocity:** `16 body parts × 3 values = 48 values`

**Total for Body Parts:** `208 values`

#### Joint Spring Observations (stiffness)
- **Spring Value:** `16 joints`

**Total for Joints:** `16 values`

**Overall Observation Size:** `224 values` (including padding if needed)

---

### Action Space

#### Torque Actions
- **Torque per Body Part:** `16 body parts × 3 directions = 48 values`

#### Spring Actions
- **Spring Adjustment per Joint:** `16 joints`

**Total Action Space Size:** `64 values`

---

### Reward System

- **Standing Reward:** The agent receives a reward based on how upright it remains. This is calculated as the dot product of the agent's `hips.up` and `Vector3.up`, encouraging the agent to maintain an upright posture.

- **Forward Movement Reward:** Although currently commented out, the reward for moving forward can be implemented as the dot product of the agent's `hips.forward` and `Vector3.forward`, promoting forward locomotion.

- **Head Position Reward:** The agent receives additional reward for keeping its head upright, measured by the dot product of the `head.up` and `Vector3.up`.

These rewards aim to incentivize the agent to stand upright and move forward, helping it learn effective walking behavior over time.

---

### Other Config
```yml
behaviors:
  walker-agent:                 # Identifier for the agent's behavior configuration
    trainer_type: ppo           # The RL algorithm to use; PPO stands for Proximal Policy Optimization

    hyperparameters:            # Parameters for the PPO algorithm
      batch_size: 2048          # Number of samples to process in each training step
      buffer_size: 20480        # Size of the buffer storing experiences for training
      learning_rate: 0.0001     # Learning rate for the optimizer
      beta: 0.001               # Weight for the entropy term in the loss function
      epsilon: 0.15             # Epsilon for the clipping function in PPO
      lambd: 0.95               # Discount factor for rewards in Generalized Advantage Estimation (GAE)
      num_epoch: 5              # Number of epochs to train on each batch

    network_settings:           # Configuration for the neural network used in the policy
      normalize: True           # Whether to normalize inputs to the network
      hidden_units: 256         # Number of hidden units per layer in the network
      num_layers: 2             # Number of layers in the network
      vis_encode_type: simple   # Type of visual encoding (e.g., simple, nature_cnn)

    reward_signals:             # Configuration for reward signals used in training
      extrinsic:                # Extrinsic reward signal configuration
        gamma: 0.99             # Discount factor for the reward signal
        strength: 1.0           # Strength of the extrinsic reward signal

    max_steps: 5000000          # Maximum number of steps to run the training
    summary_freq: 10000         # Frequency (in steps) to save training summaries

```

# ML-Agents Setup with Conda and Unity

Using Unity 2022.3 LTS

ML-Agents Release 21

### 1. Create Conda Environment
Create a new Conda environment with Python 3.9.18:

```bash
conda create -n mlagents python=3.9.18 && conda activate mlagents
```

### 2. Install PyTorch with CUDA 12.1
Install PyTorch 2.2.1 (or compatible version) with CUDA support:

```bash
pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install ML-Agents from Source
Install ML-Agents and ML-Agents Environments from the local source:

```bash
cd /path/to/ml-agents
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
```


### 4. Configure Unity for ML-Agents
- Open your Unity project.
- Go to Window > Package Manager.
- Click "+" > "Add package from disk..."
- Select **com.unity.ml-agents** and **com.unity.ml-agents.extensions** from your project root.


### 5. Verify Installation
```bash
mlagents-learn --help
```
If the help information appears, your setup is complete.

---

To train:

```bash
mlagents-learn config/walker-agent.yaml --run-id=walker-agent --force
```

To continue from a prevoius trained model:

``` bash
mlagents-learn config/walker-agent.yaml --initialize-from=walker-agent --run-id=walker-agent --force
```

