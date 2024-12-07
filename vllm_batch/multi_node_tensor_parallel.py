# Databricks notebook source
# MAGIC %pip install vllm
# MAGIC %restart_python

# COMMAND ----------

from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen2.5-7B-Instruct"

# COMMAND ----------

from vllm import LLM, SamplingParams
from vllm import LLM, SamplingParams
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

## Setup Ray on Spark (for multi-node only)
setup_ray_cluster(
  max_worker_nodes=3,
  num_cpus_worker_node=8,
  num_gpus_worker_node=1,
  num_gpus_head_node=1,
  num_cpus_head_node=8,
)

ray.init()

# COMMAND ----------

sampling_params = SamplingParams(temperature=0, max_tokens = 2048)

llm = LLM(
  model=model_name, 
  tensor_parallel_size=4, # change this to the number of GPU
  )

# COMMAND ----------

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Tell me about Singapore",
    },
]

# COMMAND ----------

outputs = llm.chat(conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)

# COMMAND ----------

print(outputs[0].outputs[0].text)

# COMMAND ----------


