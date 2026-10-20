"""Shared fakes for the offline test suite: no API keys, GPU, network or index files are needed."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeTime:
    """Replaces the `time` module inside agents.llm: waiting costs nothing and is recorded."""

    def __init__(self, now=1000.0):
        self.now = now
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedLLM:
    """Stands in for a chat client: returns scripted replies in order and records every prompt."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return types.SimpleNamespace(content=self.replies.pop(0))


@pytest.fixture
def fake_time():
    return FakeTime()


@pytest.fixture
def make_llm():
    def factory(*replies):
        return ScriptedLLM(replies)
    return factory
