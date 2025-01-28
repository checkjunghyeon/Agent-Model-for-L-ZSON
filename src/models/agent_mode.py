from enum import IntEnum

class AgentMode(IntEnum):
    SPIN = 1
    EXPLORE = 2
    EXPLOIT = 3
    NOT_USE_LLM = 4