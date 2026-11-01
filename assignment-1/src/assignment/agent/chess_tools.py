"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    # TODO(Part 3.3.b): Parse the arguments, call the provided
    # /api/simulate endpoint with fen and optional move, and return its JSON.
    # Catch any errors raised by the tool and return an error message between
    # `<chess_error></chess_error>` for the agent to address. Cover malformed
    # JSON arguments, arguments that are not an object, a missing or
    # non-string fen, a non-string move, a position or move the server rejects,
    # and a transport failure.

    try:
        args = json.loads(arguments)

        if not isinstance(args, dict):
            return "<chess_error>Arguments must be a JSON object.</chess_error>"

        fen = args.get("fen")

        if not isinstance(fen, str):
            return (
                "<chess_error>Missing or invalid fen. Expected a string.</chess_error>"
            )

        payload = {"fen": fen}

        if args.get("move") is not None:
            if not isinstance(args["move"], str):
                return (
                    "<chess_error>Invalid move format. Expected a string.</chess_error>"
                )

            payload["move"] = args["move"]

        result = _request_state(
            client,
            "POST",
            "/api/simulate",
            json=payload,
        )

        return json.dumps(result)

    except (json.JSONDecodeError, TypeError):
        return "<chess_error>Arguments must be valid JSON.</chess_error>"

    except (ValueError, RuntimeError, httpx.RequestError) as exc:
        return f"<chess_error>{exc}</chess_error>"


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    # TODO(3.1.b): Parse the arguments and POST {"move": <uci move>} to
    # /api/move. Return its JSON object. Catch any errors raised by the
    # tool and return an error message between `<chess_error></chess_error>`
    # for the agent to address. Cover malformed JSON arguments, arguments
    # that are not an object, a missing or non-string fen, a non-string move,
    # a position or move the server rejects, and a transport failure.

    try:
        args = json.loads(arguments)

        if not isinstance(args, dict):
            return "<chess_error>Arguments must be a JSON object.</chess_error>"

        if not isinstance(args.get("move"), str):
            return "<chess_error>Invalid move format.</chess_error>"

        method, endpoint = "POST", "/api/move"

        result = _request_state(client, method, endpoint, json={"move": args["move"]})

        return json.dumps(result)

    except json.JSONDecodeError:
        return "<chess_error>Arguments must be valid JSON.</chess_error>"

    except (ValueError, RuntimeError, httpx.RequestError) as exc:
        return f"<chess_error>{exc}</chess_error>"


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.

    """

    try:
        # 1. Parse tool arguments.
        args = json.loads(arguments)

        if not isinstance(args, dict):
            return "<chess_error>Arguments must be a JSON object.</chess_error>"

        # 2. Validate code.
        code = args.get("code")

        if not isinstance(code, str):
            return (
                "<chess_error>Missing or invalid code. Expected a string.</chess_error>"
            )

        # 3. Encode code to Base64.
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")

        # 4. Execute inside the remote sandbox.
        command = [
            "python",
            "/opt/assignment/sandbox_python.py",
            str(port),
            encoded_code,
        ]

        result = env.execute(command, shell=False)

        # 5. Check whether the sandbox runner itself failed.
        exception_info = result.get("exception_info", "")
        stderr = result.get("stderr", "")
        returncode = result.get("returncode", -1)

        if exception_info:
            return f"<chess_error>{exception_info}</chess_error>"

        if returncode != 0:
            message = stderr or "Sandbox execution failed."
            return f"<chess_error>{message}</chess_error>"

        # Return the runner's JSON output unchanged.
        return result.get("stdout", "")

    except json.JSONDecodeError:
        return "<chess_error>Arguments must be valid JSON.</chess_error>"

    except (TypeError, ValueError, RuntimeError) as exc:
        return f"<chess_error>{exc}</chess_error>"
    # TODO(3.4): parse the arguments and run the code in the
    # sandbox with the registered tools available by name.
    #
    # `/opt/assignment/sandbox_python.py` is a script on the `env` sandbox
    # that has access to the same tool definitions in this file. Use it to run
    # the code that the model produced as an argument to the run_python tool.
    # The script accepts two positional arguments -- `port` and a base64-encoded
    # string of code (to prevent issues with quoting). Implement this tool
    # call.
    #
    # The script prints one JSON object with `stdout`, `stderr`, and `error`
    # from running the code -- return that string as it is.
    #
    # A non-zero returncode means the sandbox itself failed, not the model's
    # code. Report `exception_info` or `stderr` as a <chess_error>.
    #
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.


def _invoke_skill(
    skills: dict[str, dict[str, str]],
    arguments: str,
) -> str:
    """Load one skill's full instructions."""

    try:
        # Parse JSON arguments.
        args = json.loads(arguments)

        if not isinstance(args, dict):
            return "<chess_error>Arguments must be a JSON object.</chess_error>"

        # Extract skill name.
        name = args.get("name")

        if not isinstance(name, str) or not name.strip():
            return "<chess_error>Missing or invalid skill name.</chess_error>"

        # Check whether the requested skill exists.
        skill = skills.get(name)

        if not isinstance(skill, dict):
            return f"<chess_error>Unknown skill: {name}</chess_error>"

        # Retrieve complete skill instructions.
        content = skill.get("content")

        if not isinstance(content, str):
            return f"<chess_error>Skill '{name}' has invalid content.</chess_error>"

        return content

    except json.JSONDecodeError:
        return "<chess_error>Arguments must be valid JSON.</chess_error>"


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
