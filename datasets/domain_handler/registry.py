# datasets/domains/registry.py
from __future__ import annotations
from typing import Dict, Type
from .base import DomainHandler

# Handlers
from .lerobot_agibot import AGIBOTLeRobotHandler
from .robomind import RobomindHandler
from .droid import DroidHandler
from .real_world import AIRAgilexHandler, AIRAgilexHQHandler, AIRBotHandler, WidowxAirHandler
from .simulations import BridgeHandler, LiberoHandler, VLABenchHandler, RobotWin2Handler, RobocasaHumanHandler, CalvinHandler, RT1Handler

# 1) Exact registry only (no heuristics)
_REGISTRY: Dict[str, Type[DomainHandler]] = {
    # LeRobot (parquet)
    "AGIBOT": AGIBOTLeRobotHandler,
    "AGIBOT-challenge": AGIBOTLeRobotHandler,

    # HDF5 (exact)
    "Calvin": CalvinHandler,
    "RT1": RT1Handler,

    # AIR family
    "AIR-AGILEX": AIRAgilexHandler,
    "AIR-AGILEX-HQ": AIRAgilexHQHandler,
    "AIRBOT": AIRBotHandler,
    "widowx-air": WidowxAirHandler,

    # Sim/others
    "Bridge": BridgeHandler,
    "libero": LiberoHandler,
    "VLABench": VLABenchHandler,
    "robotwin2_abs_ee": RobotWin2Handler,
    "robotwin2_clean": RobotWin2Handler,
    "robocasa-human": RobocasaHumanHandler,

    # Robomind
    "robomind-franka": RobomindHandler,
    "robomind-ur": RobomindHandler,
    "robomind-agilex": RobomindHandler,
    "robomind-franka-dual": RobomindHandler,

    # Droid
    "Droid-Left": DroidHandler,
    "Droid-Right": DroidHandler,
}

def get_handler_cls(dataset_name: str) -> Type[DomainHandler]:
    """Strict lookup: require explicit registration."""
    try:
        return _REGISTRY[dataset_name]
    except KeyError:
        raise KeyError(
            f"No handler registered for dataset '{dataset_name}'. "
            f"Add it to _REGISTRY in datasets/domains/registry.py."
        )
