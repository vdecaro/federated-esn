def get_template(name: str):

    if name in ["incfed", "fedip"]:
        from .vanilla import VanillaESNFederation

        return VanillaESNFederation

    elif name == "continual_fedip":
        from .continual import ContinualESNFederation

        return ContinualESNFederation

    elif name in ["ridge", "ip"]:
        from torch_esn.wrapper.vanilla import VanillaESNWrapper

        return VanillaESNWrapper

    elif name == "continual_ip":
        from torch_esn.wrapper.continual import ContinualESNWrapper

        return ContinualESNWrapper

    else:
        raise ValueError(f"Unknown template: {name}")
