# DeepTrafficQ: Reinforcement Learning for Adaptive Traffic Signal Control

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![SUMO](https://img.shields.io/badge/SUMO-Required-orange.svg)](https://www.eclipse.org/sumo/)

DeepTrafficQ is a reinforcement learning-based traffic signal control system that uses Deep Q-Networks (DQN) to minimize vehicle waiting times at 4-way intersections. This project implements a DQN agent that dynamically adjusts traffic light phases to optimize traffic flow in SUMO simulations.

## Key Features

- 🚦 **Deep Q-Learning Agent**: Uses a CNN-based neural network to learn optimal traffic signal control policies
- 🔄 **Experience Replay**: Implements memory buffer for stable training
- 📊 **SUMO Integration**: Works with SUMO traffic simulation environment
- 🏗️ **Modular Architecture**: Easy to extend to different intersection configurations

## Installation

### Prerequisites

1. Install [SUMO](https://www.eclipse.org/sumo/)
2. Python 3.8 or later

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/albinjm/DeepTrafficQ.git
   cd DeepTrafficQ
   ```

2. Install required Python packages

## Usage

1. Ensure SUMO is installed and available in your PATH
2. Load the SUMO configuration file in the SUMO GUI
3. Run the traffic control system:
   ```bash
   python traffic_light_control.py
   ```

## Project Structure

```
DeepTrafficQ/
├── traffic_light_control.py  # Main control script
├── model.py                 # DQN agent implementation
├── generator.py             # SUMO intersection generator
├── Models/                  # Saved model weights
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## DQN Architecture

The neural network consists of:
- **Input Layer**: Traffic state representation
- **Convolutional Layers**:
  - Layer 1: 16 filters (4×4), stride=2, ReLU activation
  - Layer 2: 32 filters (2×2), stride=1, ReLU activation
- **Fully Connected Layers**:
  - FC1: 128 units, ReLU activation
  - FC2: 64 units, ReLU activation
- **Output Layer**: Q-values for each possible action

## Q-Learning Implementation

The agent uses the standard Q-learning update rule:

  Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
  
  Where:
  - Q(s,a): Current Q-value for state-action pair
  - s': Next state
  - a': Next action
  - α: Learning rate (0 < α ≤ 1)
  - γ: Discount factor (0 ≤ γ < 1)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgements

- SUMO - Simulation of Urban MObility
- TensorFlow/Keras for deep learning implementation