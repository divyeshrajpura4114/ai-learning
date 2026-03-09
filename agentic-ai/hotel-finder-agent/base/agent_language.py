from typing import List

from base.goal import Goal
from base.action import Action
from base.memory import Memory
from base.environment import Environment
from base.prompt import Prompt

class AgentLanguage:
    def __init__(self):
        pass

    def construct_prompt(self,
                         actions: List[Action],
                         environment: Environment,
                         goals: List[Goal],
                         memory: Memory
                    ) -> Prompt:
        raise NotImplementedError("Subclasses must implement this method")


    def parse_response(self,
                       response: str
                    ) -> dict:
        raise NotImplementedError("Subclasses must implement this method")


