# Overview
This is a simple repo showing how you can set up vLLM and Ray to perform batch inference on Databricks.

## Notebooks
[multi_node_tensor_parallel](vllm_batch/multi_node_tensor_parallel.py): Provides an example on how to setup vLLM and Ray on a multi-node scenario

[placement_group_tensor_parallel](vllm_batch/placement_group_tensor_parallel.py): Provides an example on how to use Ray actors and placement group to only perform inference on the worker nodes

Both examples achieve the same outcome; they simply demonstrate the available options.

## Variation in LLM's Outputs
- When using vLLM, varying the ```tensor_parallel_size``` parameter may lead to slight differences in the output, which is expected. See here for [more](https://github.com/vllm-project/vllm/issues/2891)

- If you don't have acess to A10s (or A100s GPUs), you can still use T4 GPUs as well. However, note that the LLM’s outputs may vary slightly depending due to the difference in GPU architecture.