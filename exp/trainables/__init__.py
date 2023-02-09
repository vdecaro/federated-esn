def get_trainable(template: str):
    if template in ["incfed", "fedip"]:
        from .vanilla import VanillaFedESNTrainable

        return VanillaFedESNTrainable

    elif template == "continual_fedip":
        from .continual import ContinualFedESNTrainable

        return ContinualFedESNTrainable

    elif template in ["ridge", "ip"]:
        from .vanilla import VanillaESNTrainable

        return VanillaESNTrainable

    elif template == "continual_ip":
        from .continual import ContinualESNTrainable

        return ContinualESNTrainable

    else:
        raise ValueError(f"Unknown template: {template}")
