import time
import traceback
from typing import Any

from base.action import Action

class Environment:
    def execute_action(self, action: Action, args: dict) -> dict:
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return {"tool_executed": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }

    def format_result(self, result: Any) -> dict:
        return {"tool_executed": True,
                "result": result,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
            }
