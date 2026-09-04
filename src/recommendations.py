from typing import Any


def recommend_actions(
    customer: dict[str, Any],
    probability: float,
    explanation_df=None,
    threshold: float = 0.60,
) -> list[dict[str, str]]:
    """Return investigation actions with an explicit reason for each action."""
    actions: list[dict[str, str]] = []
    if probability >= threshold:
        actions.append({
            "priority": "High",
            "action": "Prioritize this customer for retention review.",
            "reason": "The predicted churn probability exceeds the review threshold.",
        })
    if customer.get("Contract") == "Month-to-month":
        actions.append({
            "priority": "Medium",
            "action": "Review annual-contract or loyalty-plan eligibility.",
            "reason": "The customer is currently on a month-to-month contract.",
        })
    if float(customer.get("tenure", 0)) <= 12:
        actions.append({
            "priority": "Medium",
            "action": "Review onboarding and early-tenure support.",
            "reason": "The customer has relatively short tenure.",
        })
    if float(customer.get("MonthlyCharges", 0)) >= 80:
        actions.append({
            "priority": "Medium",
            "action": "Review plan fit, pricing, and service value.",
            "reason": "Monthly charges are relatively high in the available dataset.",
        })
    if not actions:
        actions.append({
            "priority": "Low",
            "action": "Continue standard engagement and service-quality review.",
            "reason": "No configured high-priority retention trigger was met.",
        })
    return actions
