# WalkerAgent with Unity ML-Agents

This project involves creating and training a walker agent using Unity's ML-Agents toolkit. The walker learns to stand, walk, and balance through reinforcement learning, specifically using the Proximal Policy Optimization (PPO) algorithm.

## Table of Contents
- [Demo Overview](#Demo-Overview)
- [Agent Details](#Agent-Details)
- [Project Setup](#ML-Agents-Setup-with-Conda-and-Unity)
- [Training](#Training)

<a name="Demo-Overview"/>

## Demo Overview

#### Initial Scenario:

![Initial](https://github.com/user-attachments/assets/cb34ef31-ee08-4f09-8b13-a63e024fa21f)

#### Train to Stand: 

![standing](https://github.com/user-attachments/assets/c5ef113d-3c6a-4cd6-ae77-e64e1b1e0a4f)

#### Train to Walk Forward:

![walking](https://github.com/user-attachments/assets/8dbbd69b-03ca-4dcd-8962-2a9fe14e263a)

<br/><br/>

<a name="Agent-Details"/>

## Agent Details

### Observation Space

The agent observes the positions, rotations, velocities, and angular velocities of its body parts, as well as the spring settings of its joints.

#### Body Part Observations
- **Position:** `16 body parts × 3 coordinates = 48 values`
- **Rotation:** `16 body parts × 4 values = 64 values`
- **Velocity:** `16 body parts × 3 values = 48 values`
- **Angular Velocity:** `16 body parts × 3 values = 48 values`

**Total for Body Parts:** `208 values`

#### Joint Spring Observations (stiffness)
- **Spring Value:** `16 joints`

**Total for Joints:** `16 values`

**Overall Observation Size:** `224 values` 

---

### Action Space

The agent applies torque to body parts to control movement, and the torques are clamped for stability. The agent also applies forward and upward forces to simulate walking and stepping. Random rotations are applied at the start of each episode to vary initial conditions.

#### Torque Actions
- **Torque per Body Part:** `16 body parts × 3 directions = 48 values`

#### Spring Actions (stiffness)
- **Spring Adjustment per Joint:** `16 joints`

#### Movement Actions
- **Forward and upward movement.:** `2 joints`

**Total Action Space Size:** `66 values`

---

### Reward System

The reward system encourages the agent to Stay upright, Move forward efficiently, Alternate leg movements and Avoid falling.

- **Standing Reward:** Stay upright by maintaining a high head position.

- **Forward Movement Reward:** Move forward efficiently by rewarding forward velocity.

- **Leg movement Reward:** Alternate leg movements, simulating a walking character.

- **Falling Reward:** Avoid falling or excessive rotation by penalizing low hip or head positions.

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

<br/><br/>

<a name="ML-Agents-Setup-with-Conda-and-Unity"/>

## ML-Agents Setup with Conda and Unity

Using **Unity 2022.3 LTS**

ML-Agents **Release 21**

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

<br/><br/>

<a name="Training"/>

## Training

To train the *WalkerAgent* from scratch, run the following command:

```bash
mlagents-learn config/walker-agent.yaml --run-id=walker-agent --force
```

If you want to continue training from a previously saved model, run:

``` bash
mlagents-learn config/walker-agent.yaml --initialize-from=walker-agent --run-id=walker-agent --force
```

<br/><br/>

