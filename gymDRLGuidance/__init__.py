import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)
register(id="simpleDRLGuidance-v1", entry_point="gymDRLGuidance.envs:Environment")