from fedray.node import FedRayNode
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from typing import Dict, Type


class VirtualNode(object):
    def __init__(
        self,
        template: Type[FedRayNode],
        id: str,
        federation_id: str,
        role: str,
        config: Dict,
    ) -> None:

        self.template = template
        self.fed_id = federation_id
        self.id = id
        self.role = role
        self.config = config
        self.handle: FedRayNode = None

    def build(self, bundle_idx: int, placement_group: PlacementGroup):
        resources = placement_group.bundle_specs[bundle_idx]
        num_cpus = resources["CPU"]
        num_gpus = resources["GPU"] if "GPU" in resources else 0
        self.handle = self.template.options(
            name=self.fed_id + "/" + self.id,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group, placement_group_bundle_index=bundle_idx
            ),
        ).remote(
            node_id=self.id, role=self.role, federation_id=self.fed_id, **self.config
        )

    @property
    def built(self):
        return self.handle is not None
