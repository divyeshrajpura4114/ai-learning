from typing import Callable, List, Any, Literal

class Action:
    def __init__(self,
                name: str,
                function: Callable,
                description: str,
                parameters: dict,
                terminal: bool = False
            ):
        self.name = name
        self.function = function
        self.description = description
        self.parameters = parameters
        self.terminal = terminal

    def execute(self, **args) -> Any:
        return self.function(**args)

class ActionRegistry:
    def __init__(self):
        self.actions = {}

    def register(self, 
                action = Action
            ):
        self.actions[action.name] = action

    def get_action(self,
                   name: str
                )-> [Action, None]:
        return self.actions.get(name, None)

    def get_actions(self) -> List[Action]:
        return list(self.actions.values())

