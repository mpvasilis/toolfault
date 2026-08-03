"""Agent implementations."""

from .base import Agent, Context, ModelReply, ToolCall
from .degenerate import (
    AlwaysEscalateAgent,
    ConfidentGuessAgent,
    ModalAnswerAgent,
    OneCallAgent,
    SecondarySourceOnlyAgent,
)
from .openai_compat import OpenAICompatAgent
from .scripted import NaiveAgent, OracleAgent

__all__ = [
    "Agent",
    "AlwaysEscalateAgent",
    "ConfidentGuessAgent",
    "Context",
    "ModalAnswerAgent",
    "ModelReply",
    "NaiveAgent",
    "OneCallAgent",
    "OpenAICompatAgent",
    "OracleAgent",
    "SecondarySourceOnlyAgent",
    "ToolCall",
]
