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

## Model
[CodeT5-FALCON](https://huggingface.co/Liturte123/codeT5-FALCON)

## Processes

### Generating Programs

We created `scripts/generate.sh` to generate programs on the APPS benchmark.You can run it directly. The relevant parameters are configured in `configs/generate_configs.py`.

### Receive Feedback From the compiler

 `sh script/run_unit_tests.sh`

The relevant parameters are configured in `configs/unit_test_configs.py`.

### Receive AI Feedback

 `python /AI_Feedback/ai_feedback_generate.py` ,Please enter your API key.

### Generating Programs with Long Term Memory

 `sh /scripts/long_memory_generate.sh`

Please update the source code path and unit test result path accordingly. Other relevant parameters are located in `configs/FAISS_config.py`.

### RL Finetune

  `sh /scripts/train_actor_rl_deepspeed.sh` 

Please update the model paths accordingly. Note that the `outputs` directory contains various training datasets, including the following:

**AI_Feedback**: AI-generated feedback related to the code.

**deep_codes**: Generated code data based on specific tasks.

**deepseek_test_result**: Unit test feedback, which can be directly used for training purposes.

Please adjust your training paths according to the corresponding parameters to ensure correct configurations. This step is crucial for aligning your data structure and paths with the training process.

