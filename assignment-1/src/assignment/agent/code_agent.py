"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import (
    EXECUTE_TOOL,
    INVOKE_SKILL_TOOL,
    SEND_MESSAGE_TOOL,
)
from assignment.env import Environment


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )

        self.task = task
        self.submitted_patch = ""

        # Register the coding-agent tools.
        # INVOKE_SKILL_TOOL is registered by the base Agent when skills exist.
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        # Construct environment information.
        system_info = {
            "machine": environment.machine,
            "release": environment.release,
            "system": environment.system,
            "version": environment.version,
        }

        # Construct the system prompt.
        self.system_prompt = (
            "You are a helpful assistant working on fixing a software issue.\n"
            "The system information is as follows:\n"
            "<system_information>\n"
            f"{json.dumps(system_info, indent=2)}\n"
            "</system_information>\n"
            "Your responsibilities include:\n"
            "1. Inspecting the repository and understanding the issue.\n"
            "2. Reproducing the issue before attempting to fix it.\n"
            "3. Implementing a minimal fix.\n"
            "4. Running relevant tests to verify the fix.\n"
        )

        # Construct the task prompt.
        self.task_prompt = (
            f"The task is to fix the following software issue:\n{self.task}"
        )

        # Add skill metadata to the system prompt.
        # Full skill instructions are revealed through invoke_skill.
        if self.skills:
            metadata_str = "\n".join(
                f"Skill: {name}\nDescription: {info['metadata']}"
                for name, info in self.skills.items()
            )

            self.system_prompt += (
                "\nAvailable skills:\n"
                f"{metadata_str}\n"
                "Use the invoke_skill tool to retrieve the full instructions "
                "of a relevant skill before following its workflow."
            )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute coding-agent tools and return their observations."""

        observations: list[dict[str, str]] = []

        for tool_call in tool_calls:
            call_id = tool_call.get("id", "")

            # Validate the function object.
            function = tool_call.get("function")

            if not isinstance(function, dict):
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": (
                            "Invalid tool call: missing or invalid function object."
                        ),
                    }
                )
                continue

            tool_name = function.get("name", "")
            tool_args = function.get("arguments", "{}")

            # Parse the JSON arguments.
            try:
                parsed_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": (f"Malformed JSON in tool arguments: {tool_args}"),
                    }
                )
                continue

            # Tool arguments must be a JSON object.
            if not isinstance(parsed_args, dict):
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": ("Invalid tool arguments: expected a JSON object."),
                    }
                )
                continue

            # Execute terminal commands in the Modal environment.
            if tool_name == EXECUTE_TOOL["function"]["name"]:
                command = parsed_args.get("command", "")
                shell = parsed_args.get("shell", True)
                cwd = parsed_args.get("cwd", None)
                timeout = parsed_args.get("timeout", None)
                env = parsed_args.get("env", None)

                observation = self.env.execute(
                    command,
                    shell=shell,
                    cwd=cwd,
                    timeout=timeout,
                    env=env,
                )

                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": format_tool_output(observation),
                    }
                )

            # Send the final response to the user.
            elif tool_name == SEND_MESSAGE_TOOL["function"]["name"]:
                summary = parsed_args.get("summary", "")

                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"Message sent to user: {summary}",
                    }
                )

                self.finished = True

            # Reveal the full content of a requested skill.
            elif tool_name == INVOKE_SKILL_TOOL["function"]["name"]:
                skill_name = parsed_args.get("name", "")

                if skill_name in self.skills:
                    skill_content = self.skills[skill_name]["content"]

                    observations.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": skill_content,
                        }
                    )
                else:
                    observations.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": (
                                f"Error: Skill '{skill_name}' not found. "
                                "Available skills: "
                                f"{', '.join(self.skills.keys())}"
                            ),
                        }
                    )

            # Unknown tools are recoverable observations.
            else:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"Unknown tool: {tool_name}",
                    }
                )

        return observations
