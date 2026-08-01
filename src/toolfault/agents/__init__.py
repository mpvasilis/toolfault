"""Agent implementations."""

from .base import Agent, Context, ModelReply, ToolCall
from .openai_compat import OpenAICompatAgent
from .scripted import NaiveAgent, OracleAgent

__all__ = [
    "Agent",
    "Context",
    "ModelReply",
    "NaiveAgent",
    "OpenAICompatAgent",
    "OracleAgent",
    "ToolCall",
]
