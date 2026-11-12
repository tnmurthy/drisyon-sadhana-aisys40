"""
Drisyon Sadhana · AI System Design Starter Kit
Module 5: Agent Loop Guardrails & MCP Controller
"""

from typing import List, Dict, Any


class AgentSupervisor:
    """Controls ReAct agent loops with max step limits, cycle detection, and token budgets."""

    def __init__(self, max_steps: int = 10, max_token_budget: int = 4000):
        self.max_steps = max_steps
        self.max_token_budget = max_token_budget
        self.step_history: List[str] = []
        self.total_tokens_spent = 0

    def record_step(self, tool_name: str, tool_args_hash: str, tokens_used: int) -> Dict[str, Any]:
        """
        Records an execution step and audits for loop limits and cycle repetition.
        Returns execution status dict with 'allowed' and 'reason'.
        """
        self.total_tokens_spent += tokens_used

        # 1. Budget Guardrail
        if self.total_tokens_spent > self.max_token_budget:
            return {
                "allowed": False,
                "reason": f"Token budget exceeded ({self.total_tokens_spent} > {self.max_token_budget})"
            }

        # 2. Max Step Count Guardrail
        if len(self.step_history) >= self.max_steps:
            return {
                "allowed": False,
                "reason": f"Max steps exceeded ({len(self.step_history)} >= {self.max_steps})"
            }

        # 3. Cycle Repetition Detection (Identical tool call 3 times in a row)
        signature = f"{tool_name}:{tool_args_hash}"
        self.step_history.append(signature)

        if len(self.step_history) >= 3:
            if self.step_history[-1] == self.step_history[-2] == self.step_history[-3]:
                return {
                    "allowed": False,
                    "reason": f"Deterministic cycle detected on tool '{tool_name}'."
                }

        return {"allowed": True, "reason": "Step approved."}
