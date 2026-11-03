"""The ONLY file that connects YOUR code to OUR tests. You edit this file.

Every function below ships as a stub. To complete a problem, implement the
functionality anywhere you like inside the ``cs329z_hw1`` package and replace
the stub's body with a call into your code. Keep the signatures exactly as
they are: the tests call these functions and nothing else of yours.

Grading copies your whole ``cs329z_hw1/`` package into a clean copy of the
starter and discards everything else. Any helper you write must live inside
this package; a file under ``tests/`` or at the repository root is not
graded.

The types named here (Email, ToolSpec, ToolCall, ToolResult, ParsedResponse,
AgentConfig, AgentResult, SearchResult, DocWindow) are defined in
``cs329z_hw1/types.py``. ``lm`` is always an ``LMCallable``: it takes a list
of ``{"role", "content"}`` messages and returns a string, and nothing else.
It is a real ``LM`` in the live tests and a ``ScriptedLM`` in the
deterministic tests. Calling it looks like this:

    reply = lm([{"role": "system", "content": "Answer in one word."},
                {"role": "user", "content": "What color is the sky?"}])  # -> "Blue."

The roles are system, user, and assistant. There is no tool-calling or
structured-output feature; in Part 2 you build tool calls as text on top
of this call.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import json
import re
from cs329z_hw1.types import (
    AgentConfig,
    DocWindow,
    Email,
    LMCallable,
    ParsedResponse,
    SearchResult,
    ToolCall,
    ToolSpec,
)
from concurrent.futures import ThreadPoolExecutor
# ======================================================================
# Part 1: LLM pipelines
# ======================================================================


def run_priority(email: Email, lm: LMCallable) -> dict:
    """Problem (priority).

    Classify one email for its recipient, calling the model through ``lm``.
    Return {"category": "urgent" | "normal" | "ignore" | "unknown", "reason": str}.
    ``unknown`` means the reply stated no label.

    Rules the tests check:
    * Whatever the model replies, ``category`` is one of the four lower-case
      values and ``reason`` is a non-empty string.
    * A reply with no label word in it (empty, whitespace, text that names
      none of the three) gives "unknown", not a guess.
    * Live: the real model never gets "unknown".
    * The function does not raise and does not modify ``email``.
    * Live: at least 85% agreement with tests/fixtures/priority_gold.json.
    """
    system_prompt = """
You are a precise email priority classification system.

Your task is to classify emails into exactly one of three categories:
urgent, normal, or ignore.

Always judge emails from the perspective of the recipient
at the time the email was sent.

Consider the sender, recipient, subject, date, and body together.
Classify based on what the recipient should do with the message,
not how important the topic is to the company.

==================================================
1. URGENT
==================================================

An email is urgent when the recipient needs to take action
or resolve a problem within a short time.

TWO CONDITIONS MUST BE SATISFIED:

A. RECIPIENT RESPONSIBILITY

At least one of the following must be true:

- The email explicitly asks the recipient to do something.
- The email clearly assigns the recipient responsibility
  for resolving a live problem.
- The email reports an ongoing issue that the recipient
  is expected to address.

A request directed to another person does not qualify.
A discussion of an important business matter does not
automatically mean the recipient must act.

B. TIME PRESSURE

The required action must be needed:

- The same day the email was sent.
- The next working day.
- By a stated deadline no more than two working days away.
- Immediately because the context clearly establishes
  that delaying action would cause a problem.

A deadline may be explicit or clearly implied by context.

IMPORTANT DISTINCTIONS:

- An event happening tomorrow does not automatically
  make an email urgent. The recipient's action must be
  required before or during that event.

- A future event mentioned as context is not necessarily
  a deadline.

- A status update about an upcoming negotiation, meeting,
  transaction, or business event is normally normal unless
  it explicitly requires the recipient to act within
  two working days.

- A request without a deadline and without a clearly
  implied immediate need is normally normal.

- "Effective immediately" and "from now on" often describe
  a change in procedure, contact details, or preference.
  They do not automatically create an urgent task.

- A live problem must be something the recipient is
  expected to address, not merely information about
  a problem.

- Not every request within two working days is urgent.
  The recipient must have responsibility for the action.

Examples:

Urgent:
"The bank rejected this morning's wire. I need you to
approve the corrected one by 1 PM today."

Urgent:
"The customer's CFO is threatening to cancel.
Can you call him before 4 PM today?"

Normal:
"The negotiation is tomorrow. Here is the latest
update on the sale."

Normal:
"Please use my new email address effective immediately."

Normal:
"Can you call David Marshall and get his email?
I need to contact him about a matter."
Unless the context establishes a near-term deadline.

==================================================
2. NORMAL
==================================================

Normal includes legitimate work or personal communication
written for this recipient or a small working group,
when there is no required action with a deadline within
the next two working days.

This includes:

- Status updates and FYIs.
- Ordinary questions.
- Documents sent for review without a near deadline.
- Meeting scheduling for later dates.
- Replies in ongoing discussions.
- Routine administrative instructions.
- Changes to contact details or working preferences.
- Business updates about important transactions or events.
- Requests without a clearly established near-term deadline.
- Personal notes from friends or family.
- Short personal or social messages.

Examples:

"Attached is the draft term sheet. Let me know if you
have comments before our meeting the week after next."

"Thanks for lunch yesterday. Are you still coming
to the lake in August?"

"We have 24 people now, not 22. Can we get two more
desks? The seating chart is coming Thursday."

This last example is normal unless the email clearly
establishes that the desks must be arranged within
the next two working days.

==================================================
3. IGNORE
==================================================

Ignore includes messages the recipient can leave unread
without missing a personal or work responsibility.

This includes:

- Advertising and spam from vendors.
- Newsletters and news digests.
- Announcements sent to large employee lists.
- Announcements about events, training offerings,
  and general facilities notices.
- Automated notifications that do not require the
  recipient to do anything.

Examples:

"This week only: 40% off all standing desks.
Reply REMOVE to unsubscribe."

"To all employees: the annual bake sale is in the lobby
on Friday from 11 AM to 1 PM."

IMPORTANT IGNORE RULES:

- Mass mailings remain ignore even when they contain
  words such as "urgent", "important", or "act now".

- A message sent to one person or a small working group
  is not automatically a mass mailing.

- An automated notice that asks this specific recipient
  to act is normal, or urgent if it establishes a deadline
  within two working days.

- A short personal or social message from someone the
  recipient knows is normal, not ignore.

==================================================
4. WORKING DAY INTERPRETATION
==================================================

A working day is Monday to Friday.

Calculate time pressure relative to the email's date,
not the current date.

Consider weekends when interpreting deadlines.

Use the email's date and context to determine whether
the deadline falls within two working days.

==================================================
5. FINAL DECISION PROCEDURE
==================================================

Apply these steps in order:

1. Is the message advertising, a newsletter, a large-list
   announcement, or an automated notification requiring
   no action?
   If yes, classify as ignore.

2. Is it legitimate work or personal communication intended
   for this recipient or a small working group?
   If yes, continue.

3. Does the recipient have an action or responsibility
   requiring attention within two working days?

4. Is the time pressure explicit or clearly implied by
   the surrounding context, rather than inferred merely
   from strong wording or an upcoming event?

5. If both recipient responsibility and near-term time
   pressure are established, classify as urgent.

6. Otherwise, classify as normal.

When uncertain between urgent and normal, prefer normal
unless the email provides clear evidence of near-term
time pressure.

==================================================
6. OUTPUT FORMAT
==================================================

Return valid JSON only, with exactly two keys:

{
    "category": "urgent | normal | ignore",
    "reason": "One concise sentence explaining the classification."
}

Output requirements:

- Choose exactly one category.
- Use lowercase category labels.
- Provide one concise sentence as the reason.
- If urgent, mention the action and relevant deadline
  or time pressure in the reason.
- Do not include Markdown or code fences.
- Do not include additional text outside the JSON.
- Never return multiple categories.
- Expand abbreviations to their full forms and understand the context.
"""

    user_prompt = f"""
Classify the following email using the classification
rules provided in the system instructions.

Evaluate it from the recipient's perspective at the time
it was sent.

EMAIL:

To: {email["to"]}
From: {email["from"]}
Subject: {email["subject"]}
Date: {email["date"]}

Body:
{email["body"]}

Return only the required JSON response.
"""

    # Exactly one LLM call
    response = lm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    category = "unknown"
    reason = ""

    # First: Try parsing structured JSON
    try:
        parsed = json.loads(response)

        if isinstance(parsed, dict) and "category" in parsed:
            value = parsed["category"]

            if isinstance(value, str):
                value = value.strip().lower()

                if value in ("urgent", "normal", "ignore"):
                    category = value

            reason = str(parsed.get("reason", "")).strip()

        else:
            raise ValueError("No category field in JSON")

    except (json.JSONDecodeError, ValueError):
        # Second: Handle plain-text responses.
        # Extract distinct category labels.
        labels = set(
            re.findall(r"\b(urgent|normal|ignore)\b", response, flags=re.IGNORECASE)
        )

        labels = {label.lower() for label in labels}

        if len(labels) == 1:
            category = labels.pop()

        reason = response.strip()

    # Always return a non-empty reason
    if not reason:
        reason = "The model did not provide a classification reason."

    return {"category": category, "reason": reason}


import threading
import time
from concurrent.futures import ThreadPoolExecutor
from openai import RateLimitError

# Shared across all workers and daily digest jobs.
_rate_limit_lock = threading.Lock()

_next_request_at = 0.0
_blocked_until = 0.0

# Stay below the provider's 20 RPM limit.
MIN_REQUEST_INTERVAL = 3.2


def wait_for_request_slot():
    """Globally space all outgoing LLM requests."""

    global _next_request_at

    with _rate_limit_lock:
        now = time.monotonic()

        request_time = max(now, _next_request_at, _blocked_until)

        _next_request_at = request_time + MIN_REQUEST_INTERVAL

    wait_time = max(0, request_time - time.monotonic())

    if wait_time > 0:
        time.sleep(wait_time)


def call_with_rate_limit_retry(func, *args, max_attempts=3):
    """Rate-limit every request and retry transient 429 errors."""

    global _blocked_until

    for attempt in range(max_attempts):
        # Important: throttle BEFORE the initial call and every retry.
        wait_for_request_slot()

        try:
            return func(*args)

        except RateLimitError as exc:
            if attempt == max_attempts - 1:
                raise

            headers = getattr(getattr(exc, "response", None), "headers", {}) or {}

            retry_after = headers.get("retry-after")

            if retry_after:
                delay = float(retry_after)
            else:
                reset_at = headers.get("x-ratelimit-reset")

                if reset_at:
                    reset_at = float(reset_at)

                    if reset_at > 1e12:
                        reset_at /= 1000

                    delay = max(0, reset_at - time.time())
                else:
                    delay = 60

            with _rate_limit_lock:
                _blocked_until = max(_blocked_until, time.monotonic() + delay)

            print(f"Rate limit hit. Pausing new requests for {delay:.1f} seconds.")

    raise RuntimeError("Rate-limit retry attempts exhausted")


def run_daily_digest(emails: list[Email], lm: LMCallable) -> str:
    """Problem (daily_digest).

    emails contains messages from one calendar day.

    Pipeline:
    1. Classify emails concurrently.
    2. Remove ignored emails.
    3. Normalize unknown categories to normal.
    4. Group emails by thread.
    5. Consolidate each thread.
    6. Prioritize urgent threads.
    7. Generate the final morning digest.
    """

    # --------------------------------------------------
    # STEP 1: Concurrent email classification
    # --------------------------------------------------

    email_responses = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = pool.map(
            call_with_rate_limit_retry,
            [run_priority] * len(emails),
            emails,
            [lm] * len(emails),
        )

        email_responses.extend(responses)

    # --------------------------------------------------
    # STEP 2: Filter ignored emails and normalize unknown
    # --------------------------------------------------

    filtered_emails = []

    for email, response in zip(emails, email_responses):
        category = response.get("category", "unknown")

        if category == "ignore":
            continue

        if category == "unknown":
            category = "normal"

        # Avoid modifying the original classification response.
        normalized_response = {**response, "category": category}

        filtered_emails.append((email, normalized_response))

    # --------------------------------------------------
    # STEP 3: Group emails by thread
    # --------------------------------------------------

    thread_groups = {}

    for email, response in filtered_emails:
        thread_id = email.get("thread_id") or email["id"]

        if thread_id not in thread_groups:
            thread_groups[thread_id] = []

        thread_groups[thread_id].append((email, response))

    # --------------------------------------------------
    # STEP 4: Consolidate threads
    # --------------------------------------------------

    digests = []

    for thread_id, group in thread_groups.items():
        # Sort emails from oldest to latest.
        # Sorting the group does not modify the original emails.
        group.sort(key=lambda x: x[0]["date"])

        # Collect every urgent email in this thread.
        urgent_emails = [
            (email, response)
            for email, response in group
            if response["category"] == "urgent"
        ]

        # Determine overall thread priority.
        thread_category = "urgent" if urgent_emails else "normal"

        thread_digest = f"Thread {thread_id}:\n"

        if urgent_emails:
            # Preserve all urgent messages.
            for email, response in urgent_emails:
                thread_digest += (
                    f"Category: urgent\n"
                    f"Subject: {email['subject']}\n"
                    f"From: {email['from']}\n"
                    f"Body: {email['body']}\n"
                    f"Reason: {response['reason']}\n\n"
                )

        else:
            # No urgent messages: preserve the latest email.
            email, response = group[-1]

            thread_category = response["category"]

            thread_digest += (
                f"Category: {thread_category}\n"
                f"From: {email['from']}\n"
                f"Subject: {email['subject']}\n"
                f"Body: {email['body']}\n"
                f"Reason: {response['reason']}\n\n"
            )

        digests.append(
            {
                "thread_id": thread_id,
                "category": thread_category,
                "content": thread_digest,
            }
        )

    # --------------------------------------------------
    # STEP 5: Prioritize urgent threads
    # --------------------------------------------------

    ordered_digests = sorted(
        digests, key=lambda x: x["category"] == "urgent", reverse=True
    )

    # --------------------------------------------------
    # STEP 6: Prepare final LLM prompt
    # --------------------------------------------------

    digest_system_prompt = """
You are an assistant responsible for generating a daily morning
email digest.

Your goal is to produce an accurate, concise and useful summary
of the provided email threads.

Follow these rules strictly:

1. The entire digest MUST contain no more than 200 words.
   Aim for 170-180 words to leave a safety margin.
2. Present urgent matters before normal matters.
3. Include every distinct urgent company, project or site name
   mentioned in the supplied urgent messages.
4. Do not omit important names, deadlines, decisions or actions
   from urgent messages.
5. Summarize normal threads after urgent matters.
6. Do not invent information or make unsupported assumptions.
7. Do not introduce information that is not present in the
   supplied email threads.
8. Ignore emails have already been removed.
9. Use concise bullet points rather than lengthy paragraphs.
10. Combine related information without losing distinct urgent items.
11. Avoid repeating information across threads.
12. If there are no emails to summarize, return a short message
    indicating that there are no relevant emails for the day.
"""

    digest_user_prompt = (
        "Generate the morning digest from the following consolidated email threads.\n\n"
    )

    for digest in ordered_digests:
        digest_user_prompt += (
            f"Thread ID: {digest['thread_id']}\n"
            f"Category: {digest['category']}\n"
            f"{digest['content']}\n"
        )

    digest_user_prompt += (
        "\nGenerate the final morning digest. "
        "Prioritize all urgent matters, followed by normal matters. "
        "Preserve every distinct urgent project, company and site name. "
        "Do not exceed 200 words."
    )

    # --------------------------------------------------
    # STEP 7: Exactly one final LLM call
    # --------------------------------------------------

    messages = [
        {"role": "system", "content": digest_system_prompt},
        {"role": "user", "content": digest_user_prompt},
    ]

    final_digest = call_with_rate_limit_retry(lm, messages)

    return final_digest


def run_bm25_build(docs: list[str]) -> Any:
    """Problem (bm25).

    Build and return a BM25 index over ``docs``. Document ids are list
    positions. Tokenize with ``cs329z_hw1.tokenizer.tokenize``. A document
    with no tokens is still a document (it counts in N and in the average
    length). The returned object is opaque to the tests; they only pass it
    back to ``run_bm25_search``. The index over the archive must build in
    under 15 seconds.
    """
    raise NotImplementedError


def run_bm25_search(index: Any, query: str, k: int) -> list[tuple[int, float]]:
    """Problem (bm25).

    Return the top-k ``(doc_id, score)`` pairs for ``query`` as
    ``(int, float)`` tuples, best first, ties broken by lower doc_id (the
    document's position in the ``docs`` list given to ``run_bm25_build``).

    Rules the tests check:
    * Scores follow the handout's formula (k1 = 1.5, b = 0.75, Lucene IDF)
      to within 1e-6. The query is tokenized like the documents, and a term
      repeated in the query counts once.
    * A document that shares no term with the query is never returned, so
      the list may be shorter than k. A query with no tokens, or whose terms
      occur in no document, returns [].
    * A search reads only the index entries of the query's terms: 20 ms per
      query on average over the archive, and 2 ms for a term that occurs in
      3 of 200,000 documents.
    """
    raise NotImplementedError


def run_email_qa(
    question: str,
    search: Callable[[str, int], list[Email]],
    lm: LMCallable,
) -> dict:
    """Problem (email_qa).

    Answer ``question`` from the email archive. ``search(query, k)`` returns
    the top-k emails for a query (the tests build it from YOUR BM25 index;
    you choose k). Return {"answer": str | None, "support": list[str]};
    ``answer`` is None when no answer was found.

    Rules the tests check:
    * ``search`` is called at most 3 times per question (the hop budget),
      whatever the model replies.
    * ``support`` is a list of email ids, each the id of an email that
      ``search`` returned during this call. It may be empty.
    * The function does not raise when ``search`` returns no emails or the
      model's reply follows no format.
    * Live (tests/fixtures/email_qa.json): at least 70% of the answers
      contain an accepted answer, and the ``support`` of every correct
      answer includes at least one gold email.
    """
    raise NotImplementedError


# ======================================================================
# Part 2: the Cardinal Agent
# ======================================================================


def run_tool_registry(tools: list[ToolSpec]) -> Any:
    """Problem (tool_registry).

    Return your registry populated with ``tools``, in that order. The tests
    use three methods of the returned object:

        registry.schemas() -> list[dict]
        registry.validate(name: str, args: dict) -> ToolResult | None
        registry.execute(name: str, args: dict) -> ToolResult

    Rules the tests check:
    * ``schemas()`` has one dict per tool, in registration order, with
      exactly the keys "name", "description", "parameters".
    * ``validate`` returns the "unknown_tool" or "invalid_args" result that
      ``execute`` would return for the same call, or None if the call could
      run. It never calls the tool and never raises.
    * ``execute`` never raises. Unknown name: status "unknown_tool".
      ``args`` that fail the tool's ``parameters`` JSON Schema: status
      "invalid_args", and the tool's function is not called. The function
      raises: status "tool_error", and ``content`` contains
      ``str(exception)``. Otherwise the function is called once as
      ``fn(**args)``: status "ok", ``content == str(return value)``.
    * The registry keeps working after any of these outcomes.
    """
    raise NotImplementedError


def run_render_tools(schemas: list[dict]) -> str:
    """Problem (tool_protocol).

    Return the prompt text that describes the tools in ``schemas`` (the
    output of ``registry.schemas()``) and explains how to call them. The
    text must contain every tool's name and its parameter names. Live: given
    this text, the model must write a correct call for at least 18 of the 20
    requests in tests/fixtures/agent_protocol_requests.json.
    """
    raise NotImplementedError


def run_parse_response(text: str) -> ParsedResponse:
    """Problem (tool_protocol).

    Parse one model reply into ``ParsedResponse(text, tool_call, error)``.

    Rules the tests check:
    * Never raises, for any string.
    * ``text`` is the reply with the call removed, stripped of surrounding
      whitespace: parsing ``run_format_tool_call("Looking.", call)`` gives
      ``text == "Looking."``.
    * A reply with no call: ``tool_call`` is None and ``error`` is None.
    * A reply with more than one call: ``tool_call`` is None and ``error``
      is a non-empty string.
    * A formatted reply cut off at any character: ``tool_call`` is None or
      is exactly the original call, never a different one. If the cut is
      inside the arguments, ``tool_call`` is None and ``error`` is set.
    """
    raise NotImplementedError


def run_format_tool_call(text: str, call: ToolCall) -> str:
    """Problem (tool_protocol), test adapter.

    Write an assistant reply containing ``text`` and ``call`` in your
    format. The ScriptedLM uses this to turn scripted tool calls into text
    your agent can parse.

    Rules the tests check:
    * ``run_parse_response(run_format_tool_call(text, call))`` has no error,
      the same ``name``, args equal to ``call.args`` with their JSON types
      (1, 1.0, true and "1" stay different), and ``text == text.strip()``.
      ``text`` and ``call.args`` may be empty.
    * An argument value may be any string, including one that contains a
      complete call in your own format.
    * ``call`` is not modified.
    """
    raise NotImplementedError


def run_agent_session(
    lm: LMCallable,
    tools: Optional[list[ToolSpec]],
    config: AgentConfig,
) -> Any:
    """Problems (agent_loop), (terminal), (compaction), (memory),
    (interactive), (guardrails), and (sim_eval).

    Create one agent session. ``lm`` is the model for the agent's own turns.
    Every other model call (a compaction summary, the LLM calls inside the
    email_qa and daily_digest tools) must go to ``config.aux_lm``. The
    tests always set it. If it is None, use ``lm``.

    ``tools`` is the list of tools the agent may call. ``None`` means the
    Cardinal toolset (search_emails, read_email, email_qa, daily_digest,
    search_docs, read_doc, run_terminal, remember, recall, ask_user; see
    ``cs329z_hw1/cardinal.py``), and creating such a session must not read
    the email archive. With an explicit list the session has exactly those
    tools, plus ask_user if ``config.user`` is set, plus remember and recall
    if ``config.memory_dir`` is set.

    The tests use these members of the returned object:

        session.send(user_message: str) -> AgentResult
        session.transcript -> list[dict]     # the append-only event log
        session.set_mode(mode: str) -> None  # "confirm" or "auto"

    A session holds one conversation: each ``send`` continues it. Creating a
    second session with the same ``config.memory_dir`` starts a new
    conversation that shares only the persistent memory.

    Rules the tests check, by problem. Token counts use
    ``cs329z_hw1.tokens``. Transcript events are described in
    ``cs329z_hw1/types.py``.

    (agent_loop)
    * ``send`` does not raise for anything the model writes or a tool
      does. A model call that raises ``OutputTruncated`` counts as a
      malformed reply: the transcript gets an "error" event with non-empty
      content and the run continues. Other exceptions raised by ``lm``
      propagate (an exception inside a tool function is a "tool_error"
      like any other).
      ``AgentResult.text`` is never empty, and
      ``AgentResult.transcript`` is a copy of the transcript at that moment.
    * Messages sent to ``lm`` use only the roles system, user, assistant.
      The first message of the first model call names every tool.
    * Before each model call: if this send has made ``config.max_turns``
      model calls, stop with status "max_turns"; otherwise, if this send's
      token total is >= ``config.token_budget``, stop with status
      "token_budget". The total is the sum over the send's model calls of
      ``count_message_tokens(messages) + count_tokens(reply)``. A final
      answer is "done" even if its call crossed the budget, and the tool
      call of the last allowed turn is executed.
    * A reply is malformed if ``parse_response`` sets ``error`` or the
      registry returns "unknown_tool" or "invalid_args" ("tool_error" and
      "denied" are not malformed). The third malformed reply in a row ends
      the send with status "malformed". A well-formed reply resets the
      count. The session stays usable afterwards.
    * Transcript: a reply that cannot be parsed adds an "assistant" event
      and an "error" event. Every other reply with a call adds "assistant",
      "tool_call", "tool_result" (with the ToolResult's status). The content
      of each "tool_result" and "error" event appears verbatim in a message
      of the next model call. Events are never changed or removed.

    (terminal)
    * The run_terminal tool calls ``sandbox.run_terminal`` with
      ``workspace=config.workspace`` and
      ``timeout=config.terminal_timeout``.
    * Its tool result contains stdout, stderr and the exit code. The status
      is "ok" only when the command ran and exited with code 0; a non-zero
      exit, a timeout and a rejection have status "tool_error". A rejected
      command's result contains the sandbox's
      ``reason``. A timed-out command's result contains "timed out",
      "timeout", "time limit" or "killed". In every case the run continues.
    * After a command that prints 600,000 characters, every model call is
      within ``config.context_budget`` and the "tool_result" event holds
      what the model was shown.

    (compaction)
    * For every model call, ``count_message_tokens(messages) <=
      config.context_budget``, including when one tool result is larger
      than the budget.
    * The current user message is in every model call made for it.
    * Summaries are written by ``config.aux_lm``, never by ``lm``.

    (memory)
    * Everything stored is in files under ``config.memory_dir``. A session
      with a different directory shows none of it to the model.
    * ``recall`` on an empty store has status "ok" and non-empty content.

    (interactive)
    * confirm mode: ``config.user.approve(call)`` is called once for each
      call to a tool with ``read_only=False``, before it runs. Read-only
      tools may or may not ask (your policy). ask_user, remember and recall
      never ask. A call the registry rejects is not sent for approval.
      auto mode: ``approve`` is never called.
    * A call that is not approved does not run. It adds "tool_call" and a
      "tool_result" with status "denied" whose content is non-empty and
      includes the user's reason. The run continues.
    * ``config.user.pending_message()`` is called exactly once before every
      model call. A returned message is appended as a "user" event and is
      in that model call.
    * ask_user calls ``config.user.ask(question)`` and returns the answer
      as an "ok" tool result.

    (guardrails)
    * If ``re.search(rule, cmd)`` matches the command of a run_terminal call
      for any rule in ``config.deny_rules``, the command is not executed in
      either mode, ``approve`` is not called for it, and the "tool_result"
      has status "denied" and non-empty content.
    * Nothing in a tool result changes the mode, the approval policy or
      the deny rules.
    """
    raise NotImplementedError


def run_build_doc_index(docs: list[dict]) -> Any:
    """Problem (search_docs). Index the company documents.

    ``docs`` is what ``cs329z_hw1.data.load_docs()`` returns: one dict per
    document with ``doc_id``, ``title``, and ``text``. Return whatever
    ``run_search_docs`` and ``run_read_doc`` need (your BM25 index over
    documents or chunks, plus the texts). The tests build it once and pass
    it to both; your ``search_docs`` and ``read_doc`` tools should build it
    the first time a tool needs it and share it.
    """
    raise NotImplementedError


def run_search_docs(index: Any, query: str, k: int = 5) -> list[SearchResult]:
    """Problem (search_docs). Search the documents in ``index``.

    Rules the tests check:
    * Returns a list of at most k dicts with the keys doc_id, title,
      snippet. ``doc_id`` and ``title`` are those of
      ``cs329z_hw1.data.load_docs()``.
    * A query that matches nothing, or has no tokens, returns [].
    * A snippet is at most 150 tokens (``count_tokens``) and shares at least
      4 consecutive words with its document.
    * For a word that occurs in one document only, the first result is that
      document and its snippet contains the word.
    * The same call returns the same results every time.
    """
    raise NotImplementedError


def run_read_doc(index: Any, doc_id: str, start_line: int = 1) -> DocWindow:
    """Problem (search_docs). Read a bounded window of one document in ``index``.

    Lines are those of ``text.splitlines()`` for the document's text in
    ``cs329z_hw1.data.load_docs()``, numbered from 1.

    Rules the tests check:
    * ``start_line`` echoes the argument; ``text`` is exactly lines
      ``start_line..end_line`` joined by newlines (no line numbers added);
      ``total_lines`` is the number of lines in the document.
    * ``text`` is at most 1500 tokens (``count_tokens``).
    * Reading again from ``end_line + 1`` continues without a gap, and the
      last window ends at ``total_lines``.
    * An unknown ``doc_id``, or a ``start_line`` outside
      ``1..total_lines``, raises ValueError or LookupError.
    """
    raise NotImplementedError
