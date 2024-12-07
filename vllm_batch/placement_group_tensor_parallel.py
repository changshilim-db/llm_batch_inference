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
from ray.util.placement_group import placement_group, placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

setup_ray_cluster(
  max_worker_nodes=2,
  num_cpus_worker_node=8,
  num_gpus_worker_node=1,
  num_gpus_head_node=0,
  num_cpus_head_node=4,
)

ray.init()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Placement Group

# COMMAND ----------

pg = placement_group([{"CPU": 8, "GPU": 1}, {"CPU": 8, "GPU": 1}])
ready, unready = ray.wait([pg.ready()], timeout=10)

print(placement_group_table(pg))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Ray Actors

# COMMAND ----------

@ray.remote(num_cpus=8, num_gpus=1)
class vllmActor:
    def __init__(
        self, model_name, max_model_len=2048, tensor_parallel_size=1, dtype="auto"
    ):
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
        )

    def _get_output_string(self, outputs):
        output_str = ""
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            output_str += f"{generated_text}\n"
        return output_str

    def batch_inference(self, inputs, sampling_params):
        outputs = self.llm.chat(inputs, sampling_params=sampling_params, use_tqdm=False)
        output_str = self._get_output_string(outputs)
        return output_str

# COMMAND ----------

sampling_params = SamplingParams(temperature=0, max_tokens = 2048)
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

inputs = [conversation for _ in range(10)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute on Placement Groups

# COMMAND ----------

num_actors = 2

actors = [
    vllmActor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
    ).remote(model_name=model_name)
    for _ in range(num_actors)
]

# COMMAND ----------

results = ray.get(
    [
        ray_actor.batch_inference.remote(inputs=inputs, sampling_params=sampling_params)
        for ray_actor in actors
    ]
)

# COMMAND ----------

print(results[0])

# COMMAND ----------

print(results[1])

# COMMAND ----------


