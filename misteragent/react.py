from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import ast
import sys
from io import StringIO

from misteragent.prompts.code_agent import system_prompt

class StepType(Enum):
    THOUGHT = "THOUGHT"
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"
    FINAL_ANSWER = "FINAL_ANSWER"

@dataclass
class ReActStep:
    type: StepType
    content: str

class CodeExecutionError(Exception):
    """Exception raised when there is an error executing generated code"""
    pass

class ReActAgent:
    """
    Agent class that solves tasks step by step using the ReAct framework:
    The agent follows a cycle of:
    1. Reasoning about the current state and what to do next, and generating the corresponding code
    2. Executing the code and observing the results
    3. Updating the state based on the results
    
    This implementation specializes the Action step to execute Python code,
    allowing the agent to solve programming and data manipulation tasks.
    """

    def __init__(self, llm_client, max_steps: int = 10):
        """
        Initialize ReAct agent
        
        Args:
            llm_client: Client to interact with the LLM
            max_steps: Maximum number of reasoning steps before stopping
        """
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.history: List[ReActStep] = []
        self.local_vars: Dict[str, Any] = {}
        
    def _add_step(self, step_type: StepType, content: str) -> None:
        """
        Internal method to add a new step to the reasoning chain
        
        Args:
            step_type: Type of the reasoning step
            content: Content of the step
        """
        self.history.append(ReActStep(type=step_type, content=content))
        
    def _format_history(self) -> str:
        """Format the reasoning history into a string (internal method)"""
        formatted = []
        for step in self.history:
            formatted.append(f"{step.type.value}:\n{step.content}\n")
        return "\n".join(formatted)

    def _reason(self, task: str, context: str) -> tuple[str, str | None]:
        """
        Internal method for the reasoning phase: Get the LLM to think about the next step 
        and potentially generate code
        
        Args:
            task: The original task description
            context: Current context including history
            
        Returns:
            tuple containing:
                - thought: The reasoning process
                - code: The code to execute (None if no action needed)
        """
        response = self.llm_client.get_completion(context)
        
        if "ACTION:" in response:
            thought = response.split("ACTION:")[0].strip()
            code = response.split("ACTION:")[1].strip()
            return thought, code
        
        return response, None

    def _act(self, code: str) -> str:
        """
        Internal method for the acting phase: Execute the generated code and observe results
        
        Args:
            code: Python code to execute
            
        Returns:
            Observation from code execution
        """
        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        try:
            # Parse the code first to catch syntax errors
            ast.parse(code)
            
            # Execute the code in a controlled environment
            exec(code, {}, self.local_vars)
            output = redirected_output.getvalue()
            return output if output else "Code executed successfully with no output"
            
        except Exception as e:
            return f"Error executing code: {str(e)}"
        
        finally:
            sys.stdout = old_stdout
    
    def run(self, task: str) -> str:
        """
        Execute the ReAct reasoning loop
        
        Args:
            task: The task description to solve
            
        Returns:
            Final answer or solution to the task
        """
        prompt = (
            f"Task: {task}\n\n"
            "Solve this step by step:\n"
            "1. Think about the solution\n"
            "2. Generate Python code to implement it\n"
            "3. Observe the results and iterate if needed\n\n"
            "Use these markers:\n"
            "THOUGHT: for explaining your reasoning\n"
            "ACTION: for Python code to execute\n"
            "FINAL ANSWER: when you have solved the task"
        )
        
        for _ in range(self.max_steps):
            current_context = f"{prompt}\n\n{self._format_history()}"
            
            # Reasoning phase
            thought, code = self._reason(task, current_context)
            
            if "FINAL ANSWER:" in thought:
                self._add_step(StepType.FINAL_ANSWER, thought)
                return thought
            
            self._add_step(StepType.THOUGHT, thought)
            
            # Acting phase (if code was generated)
            if code is not None:
                self._add_step(StepType.ACTION, code)
                observation = self._act(code)
                self._add_step(StepType.OBSERVATION, observation)
                
        return "Maximum steps reached without finding solution"
