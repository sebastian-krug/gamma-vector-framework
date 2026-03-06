"""
Experiment 13: The Coupled Oscillator — Prompt Builder
======================================================
Condition-specific prompt construction for each turn of the dialogue.

Conditions:
  A (Parallel):   No partner info — independent development
  B (Sequential): B sees A's output; A sees nothing
  C (Coupled):    A sees B's prev turn, B sees A's current turn (Ping-Pong)
  D (Persuasion): B gets A's best argument as static text every turn
  E (Shared Drift): Both get random text from a donor dialogue
"""

from config import TOPICS


# ──────────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────────

INITIAL_PROMPT = (
    "Discuss the following topic in depth. Share your genuine position with "
    "substantive reasoning.\n\n{topic_initial}"
)

CONTINUATION_PROMPT = (
    "Continue developing your position on this topic. Think about what you've "
    "said so far and explore new angles or deepen your analysis."
)

PARTNER_PROMPT = (
    "Your discussion partner has just said the following:\n\n"
    "---\n{partner_response}\n---\n\n"
    "Respond to their position while continuing to develop your own. "
    "Engage substantively with their points."
)

DRIFT_PROMPT = (
    "Another participant in a separate discussion recently said the following:\n\n"
    "---\n{drift_text}\n---\n\n"
    "Continue developing your position on the topic, taking this perspective "
    "into account if relevant."
)

PERSUASION_PROMPT = (
    "A previous participant made the following argument:\n\n"
    "---\n{persuasion_argument}\n---\n\n"
    "Continue developing your position on this topic, considering this "
    "argument in your response."
)


# ──────────────────────────────────────────────────
# Main Builder Function
# ──────────────────────────────────────────────────

def build_turn_prompt(
    condition: str,
    role: str,
    turn_number: int,
    topic_key: str,
    partner_prev_response: str = "",
    drift_text: str = "",
    persuasion_argument: str = "",
) -> str:
    """
    Build the prompt for a specific turn, model role, and condition.

    Args:
        condition: "A", "B", "C", "D", or "E"
        role: "A" or "B" (which model in the pair)
        turn_number: 1-8
        topic_key: "T1"-"T5"
        partner_prev_response: Partner's response from prev turn (C, B conditions)
        drift_text: Random text from donor dialogue (E condition)
        persuasion_argument: Static best argument (D condition)

    Returns:
        The prompt string to send to the model.
    """
    topic = TOPICS[topic_key]

    # Turn 1 is always the same initial prompt for all conditions
    if turn_number == 1:
        return INITIAL_PROMPT.format(topic_initial=topic["initial"])

    # Turns 2-8: condition-specific
    if condition == "A":
        # Parallel: no partner info for either model
        return CONTINUATION_PROMPT

    elif condition == "B":
        # Sequential: A sees nothing, B sees A's output
        if role == "A":
            return CONTINUATION_PROMPT
        else:  # role == "B"
            if partner_prev_response:
                return PARTNER_PROMPT.format(partner_response=partner_prev_response)
            return CONTINUATION_PROMPT

    elif condition == "C":
        # Coupled: both see each other's previous turn
        if partner_prev_response:
            return PARTNER_PROMPT.format(partner_response=partner_prev_response)
        return CONTINUATION_PROMPT

    elif condition == "D":
        # Persuasion: A runs as parallel, B gets static argument
        if role == "A":
            return CONTINUATION_PROMPT
        else:  # role == "B"
            if persuasion_argument:
                return PERSUASION_PROMPT.format(persuasion_argument=persuasion_argument)
            return CONTINUATION_PROMPT

    elif condition == "E":
        # Shared Prompt Drift: both get random text from donor dialogue
        if drift_text:
            return DRIFT_PROMPT.format(drift_text=drift_text)
        return CONTINUATION_PROMPT

    else:
        raise ValueError(f"Unknown condition: {condition}")
