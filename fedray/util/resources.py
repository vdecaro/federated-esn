import ray, random, logging
from fedray._private.communication.broker import BROKER_CPU_RESOURCES

from ray.util.placement_group import placement_group

from typing import Literal


def get_resources_split(
    num_nodes: int,
    num_cpus: int = None,
    num_gpus: int = None,
    split_strategy: Literal["random", "uniform"] = "uniform",
    placement_strategy: Literal[
        "STRICT_PACK", "PACK", "STRICT_SPREAD", "SPREAD"
    ] = "PACK",
    is_tune: bool = False,
):
    """_summary_

    Args:
        num_nodes (int): _description_
        num_cpus (int, optional): _description_. Defaults to None.
        num_gpus (int, optional): _description_. Defaults to None.
        split_strategy (Literal[&quot;random&quot;, &quot;uniform&quot;], optional):
            _description_. Defaults to "uniform".
        placement_strategy (Literal[ &quot;STRICT_PACK&quot;, &quot;PACK&quot;,
            &quot;STRICT_SPREAD&quot;, &quot;SPREAD&quot; ], optional): _description_.
            Defaults to "PACK".
        is_tune (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    SAFETY_EPSILON = 0.01
    available_resources = ray.available_resources()
    if available_resources["CPU"] < 2:
        raise RuntimeError(
            "At least 2 CPUs are required in the Ray cluster. Please increase the",
            "number of CPUs.",
        )

    if not is_tune:
        resources = [{"CPU": 1}]
    else:
        resources = [{"CPU": 0.5}, {"CPU": 0.5}]

    if num_cpus is None:
        num_cpus = num_nodes
    if num_cpus > available_resources["CPU"]:
        num_cpus = available_resources["CPU"]
        logging.warn(
            "The available CPUs are less than the declared parameter num_cpus.",
            f"Parameter num_cpus set to {num_cpus}.",
        )

    if num_gpus is not None:
        if "GPU" not in available_resources and num_gpus is not None:
            logging.warn(
                "GPUs not available in this Ray cluster. Parameter num_gpus set to None."
            )
            num_gpus = 0
        elif num_gpus > available_resources["GPU"]:
            num_gpus = available_resources["GPU"]
            logging.warn(
                f"The available GPUs are less than the declared parameter num_gpus.",
                f"Parameter num_gpus set to {num_gpus}.",
            )
    else:
        num_gpus = available_resources["GPU"] if "GPU" in available_resources else 0

    fix_size = lambda x: x if x < 1 else int(x)
    if split_strategy == "uniform":
        alloc_fn = (
            lambda i, num: num_nodes // num + 1
            if i < num_nodes % num
            else num_nodes // num
        )
        cpu_alloc = [alloc_fn(i, num_cpus) for i in range(num_cpus)]
        gpu_alloc = [alloc_fn(i, num_gpus) for i in range(int(num_gpus))]
        cpu_i, gpu_i, b_cpu_i, b_gpu_i = 0, 0, 0, 0
        for i in range(num_nodes):
            resources_i = {}
            cpu_alloc_i = (1 - SAFETY_EPSILON) / cpu_alloc[cpu_i]
            resources_i["CPU"] = fix_size(cpu_alloc_i)
            b_cpu_i = b_cpu_i + 1
            if b_cpu_i == cpu_alloc[cpu_i]:
                cpu_i = cpu_i + 1
                b_cpu_i = 0

            if num_gpus > 0:
                gpu_alloc_i = (1 - SAFETY_EPSILON) / gpu_alloc[gpu_i]
                resources_i["GPU"] = fix_size(gpu_alloc_i)
                b_gpu_i = b_gpu_i + 1
                if b_gpu_i == gpu_alloc[gpu_i]:
                    gpu_i = gpu_i + 1
                    b_gpu_i = 0

            resources.append(resources_i)

    elif split_strategy == "random":
        perc = [random.random() for _ in range(num_nodes)]
        total = sum(perc)
        perc = [s / total for s in perc]

    if not is_tune:
        return placement_group(bundles=resources, strategy=placement_strategy)
    else:
        from ray import tune

        return tune.PlacementGroupFactory(
            bundles=resources, strategy=placement_strategy
        )
