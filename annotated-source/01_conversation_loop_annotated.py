"""Agent 对话主循环 — Hermes 的核心引擎。

═══════════════════════════════════════════════════════════════════
中文说明 (给 Python 新人)
═══════════════════════════════════════════════════════════════════

这是 Hermes Agent 最重要的一个文件, 整个 Agent 的"思考 - 行动"循环
都在这里. 整个文件 5800+ 行, 但 99% 的逻辑都集中在一个函数:

    run_conversation(agent, user_message, ...) -> Dict

它做的事情 (简化版):
    1. 用户输入 → 大模型思考
    2. 大模型决定:
       - 如果要调工具 (比如查文件, 跑命令) → 执行工具 → 把结果告诉大模型
       - 如果不需要工具 → 输出回复 → 结束
    3. 重复 1-2 直到完成 (受 max_iterations 和 token 预算限制)

═══════════════════════════════════════════════════════════════════
英文原版 (原注释保留,方便对照)
═══════════════════════════════════════════════════════════════════

The agent conversation loop — extracted from ``run_agent.AIAgent``.

This is the biggest single chunk pulled out of ``run_agent.py``: the
roughly 3,900-line :func:`run_conversation` body that drives one user
turn through the agent (model call, tool dispatch, retries, fallbacks,
compression, post-turn hooks, background memory/skill review nudges).

The function takes the parent ``AIAgent`` instance as its first
argument (``agent``) and accesses its state via attribute lookup.
``_ra().AIAgent.run_conversation`` is now a thin forwarder.

Symbols that production code or tests patch on ``run_agent`` directly
(``handle_function_call``, ``_set_interrupt``, ``OpenAI``, ...) are
resolved through :func:`_ra` so those patches keep working.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import ssl
import sys
import time
from typing import Any, Dict, List, Optional

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.conversation_compression import conversation_history_after_compression
from agent.display import KawaiiSpinner
from agent.error_classifier import FailoverReason, classify_api_error
from agent.iteration_budget import IterationBudget
from agent.turn_context import (
    build_turn_context,
    compose_user_api_content,
    reanchor_current_turn_user_idx,
)
from agent.turn_retry_state import TurnRetryState
from agent.message_sanitization import (
    close_interrupted_tool_sequence,
    _repair_tool_call_arguments,
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_surrogates,
    _sanitize_tools_non_ascii,
    _strip_images_from_messages,