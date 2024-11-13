# FALCON: Feedback-driven Adaptive Long/short-term memory reinforced Coding Optimization

## FALCON Overview
![overview](https://github.com/liturte/FALCON/blob/main/Data/overview%20(1).jpg)

we explore the FALCON framework, which integrates comprehensive unit testing with reinforcement learning, supported by both long-term and short-term memory buffers. During the code generation process, the system stores task descriptions, generated code, and various feedback (e.g., compilation results, code style, and complexity) in the long-term memory buffer. By retrieving this information, the model references high-quality code, avoids past mistakes, and ensures adherence to required standards. After generating the code, a judge model evaluates it and calculates rewards based on the feedback, which are then used to update the model's parameters through reinforcement learning. All generated code and feedback are stored for future reference and optimization. The combination of long-term and short-term memory feedback in the FALCON framework allows the model to not only learn from a wide range of historical data but also adapt quickly to new tasks based on recent performance.

## Installation

The code requires some dependencies as specified in `requirements.txt`. Please follow the relevant libraries to install or run:

```
pip install -r requirements.txt
```

## Datasets

**APPS**: Please follow the downloading and preprocessing instructions provided [here]([hendrycks/apps: APPS: Automated Programming Progress Standard (NeurIPS 2021)](https://github.com/hendrycks/apps))

Download and unzip all files into the data folder.

## Processes

### Generating Programs

- **CodeT5**: python script/generate_online_parallel.py
- **Deepseek-Coder**: python script/generate_Deepseek-Coder_online_parallel.py

### Generating Feedback 

python Feedback/program_feedback.py

### RL Finetune

- **CodeT5**: sh script/train_actor_rl_online_v1_deepspeed.sh
- **Deepseek-Coder**: sh script/train_actor_rl_Deepseek-Coder_online_v1_deepspeed.sh

### Run Unit Test：

- sh script/run_unit_tests.sh

### Compute pass@k：

- python compute_pass_at_k_metric.py
