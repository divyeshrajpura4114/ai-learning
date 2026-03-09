import os
import json
from dataclasses import dataclass, field
from typing import List, Callable, Any

from base.goal import Goal
from base.action import Action, ActionRegistry
from base.memory import Memory
from base.environment import Environment
from base.prompt import Prompt
from base.agent_language import AgentLanguage

class AgentFunctionCallingActionLanguage(AgentLanguage):

    def __init__(self):
        super().__init__()

    def format_goals(self, goals: List[Goal]) -> List:
        # Map all goals to a single string that concatenates their description
        # and combine into a single message of type system
        sep = "\n-------------------\n"
        goal_instructions = "\n\n".join([f"{goal.name}:{sep}{goal.description}{sep}" for goal in goals])
        return [
            {"role": "system", "content": goal_instructions}
        ]

    def format_memory(self, memory: Memory) -> List:
        """Generate response from language model"""
        # Map all environment results to a role:user messages
        # Map all assistant messages to a role:assistant messages
        # Map all user messages to a role:user messages
        items = memory.get_memories()
        mapped_items = []
        for item in items:

            content = item.get("content", None)
            if not content:
                content = json.dumps(item, indent=4)

            if item["type"] == "assistant":
                mapped_items.append({"role": "assistant", "content": content})
            elif item["type"] == "environment":
                mapped_items.append({"role": "assistant", "content": content})
            else:
                mapped_items.append({"role": "user", "content": content})

        return mapped_items

    def format_actions(self, actions: List[Action]) -> [List,List]:
        """Generate response from language model"""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    # Include up to 1024 characters of the description
                    "description": action.description[:1024],
                    "parameters": action.parameters,
                },
            } for action in actions
        ]

        return tools

    def construct_prompt(self,
                         actions: List[Action],
                         environment: Environment,
                         goals: List[Goal],
                         memory: Memory) -> Prompt:

        prompt = []
        prompt += self.format_goals(goals)
        prompt += self.format_memory(memory)

        tools = self.format_actions(actions)

        return Prompt(messages=prompt, tools=tools)

    def adapt_prompt_after_parsing_error(self,
                                         prompt: Prompt,
                                         response: str,
                                         traceback: str,
                                         error: Any,
                                         retries_left: int) -> Prompt:

        return prompt

    def parse_response(self, response: str) -> dict:
        """Parse LLM response into structured format by extracting the ```json block"""

        try:
            return json.loads(response)

        except Exception as e:
            return {
                "tool": "terminate",
                "args": {"message":response}
            }

class Agent:
    def __init__(self,
                goals: List[Goal],
                agent_language: AgentLanguage,
                action_registry: ActionRegistry,
                generate_response: Callable[[Prompt], str],
                environment: Environment
            ):
    
        self.goals = goals
        self.agent_language = agent_language
        self.actions = action_registry
        self.generate_response = generate_response
        self.environment = environment

    def construct_prompt(self,
                        goals: List[Goal],
                        memory: Memory,
                        actions: ActionRegistry
                    ) -> Prompt:
        return self.agent_language.construct_prompt(actions = actions.get_actions(),
                                                    environment = Environment,
                                                    goals = goals,
                                                    memory = memory
                                                )

    def get_action(self,
                   response
                ):
        invocation = self.agent_language.parse_response(response)
        action = self.actions.get_action(invocation["tool"])
        return action, invocation

    def should_terminate(self,
                        response: str
                    ):
        action_def, _ = self.get_action(response)
        return action_def.terminal

    def set_current_task(self, 
                        memory: Memory,
                        task: str
                    ): 
        memory.add_memory({"type": "user", "content": task})

    def update_memory(self,
                        memory: Memory,
                        response: str,
                        result: dict
                    ):
        new_memories = [{"type": "assistant", "content": response,
                        "type": "environment", "content": json.dumps(result)
                        }]
        for m in new_memories:
            memory.add_memory(m)

    def prompt_llm_for_action(self,
                                full_prompt: Prompt
                            ) -> str:
        response = self.generate_response(full_prompt)
        return response

    def run(self,
            user_input: str,
            memory = None,
            max_iterations: int = 10
        ) -> Memory:
        memory = memory or Memory()
        self.set_current_task(memory = memory, 
                                task = user_input
                            )
        for iteration in range(max_iterations):
            prompt = self.construct_prompt(goals = self.goals,
                                        memory = memory,
                                        actions = self.actions
                                        )

            response = self.prompt_llm_for_action(full_prompt = prompt)

            action, invocation = self.get_action(response = response)

            result = self.environment.execute_action(action,
                                                    invocation["args"]
                                                )

            self.update_memory(memory = memory,
                            response = response,
                            result = result
                            )
            if self.should_terminate(response):
                break
        return memory
