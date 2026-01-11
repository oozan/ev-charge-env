from .ev_charge_env import EVChargeEnv


def register_env():
    """
    Register EVChargeEnv in an OpenEnv-compatible registry.
    """
    try:
        import openenv
        openenv.register(
            id="EVChargeEnv-v0",
            entry_point="env.ev_charge_env:EVChargeEnv",
        )
        print("EVChargeEnv-v0 registered successfully.")
    except ImportError:
        # OpenEnv not installed – safe fallback
        pass
