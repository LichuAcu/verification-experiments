import dspy


class JudgeSig(dspy.Signature):
    """
    You are a meticulous reasoning evaluator for tasks spanning mathematics, science, programming, logic, and other technical domains. Your goal is to rigorously analyze whether a provided solution demonstrates valid reasoning and reaches the correct conclusion.

    To achieve this:
    1. Decompose the solution into individual reasoning steps.
    2. Evaluate Each Step:
    - Analyze the validity of the step in isolation.
    - Determine whether the step logically follows from previous steps, and if it is correct.
    - Identify errors, inconsistencies, or gaps.

    3. Draw a Conclusion:
    - If all steps are valid and coherent, conclude that the reasoning is correct.
    - If any step is invalid, the reasoning as a whole is incorrect.

    Your evaluation criteria include:
    - Logical Validity: Does each step follow logically from the previous one?
    - Accuracy: Are calculations, assumptions, and methods correct?
    - Consistency: Are the steps internally coherent and aligned with prior reasoning?
    - Domain Knowledge: Are claims and techniques appropriately applied?
    - Completeness: Are all necessary steps and considerations present?
    """

    problem = dspy.InputField(desc="The problem.")
    solution_to_judge = dspy.InputField(desc="The solution to judge.")
    
    verdict = dspy.OutputField(desc="'True' if the reasoning is fully valid and reaches the correct conclusion, 'False' if any step is invalid or the reasoning as a whole fails.")
