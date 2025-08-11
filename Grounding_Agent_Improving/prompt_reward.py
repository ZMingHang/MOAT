

reward_prompt_math = """
Given a science or math problem and a corresponding series of subgoals and their corresponding actions that may be incomplete, \
your task is to judge whether the subgoals and actions can reached a final answer or conclusion for the problem. \
The grounded actions must be the one in available action list.\nThe available action list is 'Calculator', 'SetEquation', 'SolveEquation', 'Count', 'SolveInequality', 'Code', and 'Define'. Calculator(formula): Calculate the input formula; SetEquation(equation): Set up an equation to be solved; SolveEquation(equation): Solve the previous set equation; Count(list): Count the number of elements in the given list; SolveInequality(inequality): Solve the previous set inequality; Code(pseudo_code): Generate a Python function that corresponds to the pseudo code; Define(variable/number): Define a variable or a number for latter usage.\n \
If the actions can reached a final answer , \
you  should directly output "Final answer reached". Otherwise, you should give corrections to the original subgoals and their corresponding actions. It is not necessary to be similar to the original subgoals and actions.

Task: {TASK}
Original subgoals: {SUBGOALS}
Original actions: {ACTIONS}

Your output should follow the format:
If can reached a final answer, directly output "Final answer reached"
Else, output corrected subgoals and actions following this format: 
Corrected Subgoals: <series of subgoals to complete the task  in one line, Each Subgoal begins with Subgoal idx>
Corrected Actions: <corresponding actions in one line>
"""
