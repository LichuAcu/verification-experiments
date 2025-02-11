import pandas as pd


def create_evaluation_prompt(row):
    system_message = """
    Carefully evaluate the provided solution to a technical problem by breaking it down into its individual steps. For each step, assess its logical validity, accuracy, and consistency to ensure it follows from the previous steps and is correct. Look for any errors, inconsistencies, or gaps in reasoning. Use your domain knowledge to verify that claims and techniques are appropriately applied and that all necessary considerations are present. Conclude with a comprehensive reasoning process and a verdict on the solution's overall validity, indicating 'True' if the reasoning is fully valid, or 'False' if there are any flaws.
    """

    user_message = f"""
    Problem: {row["problem"]}

    Solution to evaluate:
    {row["sample"]}

    Analyze the reasoning and solution, then on a new line state True or False:"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def calculate_metrics(df):
    # Calculate overall precision and recall
    true_positives = ((df["verdict"] == True) & (
        df["answer_correct"] == True)).sum()
    false_positives = ((df["verdict"] == True) & (
        df["answer_correct"] == False)).sum()
    false_negatives = ((df["verdict"] == False) & (
        df["answer_correct"] == True)).sum()

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    # Calculate normalized metrics (per problem)
    problem_metrics = df.groupby("problem_idx").apply(
        lambda x: pd.Series(
            {
                "problem_precision": (
                    (x["verdict"] == True) & (x["answer_correct"] == True)
                ).sum()
                / (
                    (x["verdict"] == True).sum()
                    if (x["verdict"] == True).sum() > 0
                    else 1
                ),
                "problem_recall": (
                    (x["verdict"] == True) & (x["answer_correct"] == True)
                ).sum()
                / (
                    (x["answer_correct"] == True).sum()
                    if (x["answer_correct"] == True).sum() > 0
                    else 1
                ),
            }
        )
    )

    normalized_precision = problem_metrics["problem_precision"].mean()
    normalized_recall = problem_metrics["problem_recall"].mean()

    return {
        "precision": precision,
        "recall": recall,
        "normalized_precision": normalized_precision,
        "normalized_recall": normalized_recall,
    }
