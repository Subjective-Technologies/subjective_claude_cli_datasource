from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from brainboost_data_source_logger_package.BBLogger import BBLogger
from subjective_abstract_data_source_package import SubjectiveDataSource


class SubjectiveClaudeCliDataSource(SubjectiveDataSource):
    """On-demand, resumable chat datasource backed by the local Claude Code CLI.

    Authentication stays owned by Claude CLI. By default this datasource reuses
    an existing ``claude auth login`` session. An API key can be supplied
    explicitly, but it is never copied into global process state or returned
    to callers.
    """

    AUTH_EXISTING_SESSION = "existing_session"
    AUTH_API_KEY = "api_key"
    MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
    PERMISSION_MODES = (
        "default",
        "acceptEdits",
        "auto",
        "bypassPermissions",
        "dontAsk",
        "plan",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        connection = self._connection or {}

        # In-flight turn, for cancel(). A fresh instance is built per turn (see
        # datasources.run), so this is one process at a time, not a registry.
        self._active_process: subprocess.Popen | None = None
        self._cancelled = False
        self._process_lock = threading.Lock()

        auth_method = str(connection.get("auth_method") or self.AUTH_EXISTING_SESSION)
        # Older saved connections may have called CLI OAuth mode simply "oauth".
        self.auth_method = (
            self.AUTH_EXISTING_SESSION if auth_method == "oauth" else auth_method
        )
        self.api_key = str(connection.get("api_key") or "")
        self.configured_claude_path = str(connection.get("claude_path") or "")
        self.model = str(connection.get("model") or "")
        self.permission_mode = str(connection.get("permission_mode") or "default")
        self.working_directory = str(
            connection.get("working_directory") or os.getcwd()
        )
        self.timeout = self._coerce_int(connection.get("timeout"), 600, minimum=10)
        self.max_turns = self._coerce_optional_int(connection.get("max_turns"), minimum=1)
        self.allowed_tools = str(connection.get("allowed_tools") or "").strip()
        self.system_prompt = str(connection.get("system_prompt") or "").strip()
        self.append_system_prompt = str(
            connection.get("append_system_prompt") or ""
        ).strip()
        self.bare = self._coerce_bool(connection.get("bare", False))
        self.dangerously_skip_permissions = self._coerce_bool(
            connection.get("dangerously_skip_permissions", False)
        )
        self.resume_session = self._coerce_bool(
            connection.get("resume_session", True)
        )
        self.session_id = str(connection.get("session_id") or "") or None

        self._claude_path: str | None = None
        self._lock = threading.RLock()

    @classmethod
    def connection_schema(cls) -> dict:
        return {
            "auth_method": {
                "type": "select",
                "label": "Authentication",
                "required": True,
                "default": cls.AUTH_EXISTING_SESSION,
                "options": [cls.AUTH_EXISTING_SESSION, cls.AUTH_API_KEY],
                "description": (
                    "Reuse existing Claude CLI login, or provide an API key. "
                    "Run 'claude auth login' separately when login is required."
                ),
            },
            "api_key": {
                "type": "password",
                "label": "Anthropic API Key",
                "required": False,
                "placeholder": "sk-ant-...",
                "description": "Used only when Authentication is api_key.",
            },
            "claude_path": {
                "type": "file_path",
                "label": "Claude CLI Path",
                "required": False,
                "placeholder": "/path/to/claude",
                "description": "Optional override. PATH is searched when blank.",
            },
            "model": {
                "type": "text",
                "label": "Model",
                "required": False,
                "placeholder": "Leave blank to use Claude CLI default (e.g. sonnet)",
            },
            "permission_mode": {
                "type": "select",
                "label": "Permission Mode",
                "options": list(cls.PERMISSION_MODES),
                "default": "default",
                "description": (
                    "How Claude Code handles tool permissions. Keep 'default' "
                    "for remote/mobile use unless workspace policy is understood."
                ),
            },
            "working_directory": {
                "type": "folder_path",
                "label": "Working Directory",
                "placeholder": "/path/to/project",
                "description": "Workspace presented to a new Claude session.",
            },
            "session_id": {
                "type": "text",
                "label": "Initial Session ID",
                "required": False,
                "description": "Optional Claude session UUID to resume.",
            },
            "resume_session": {
                "type": "checkbox",
                "label": "Resume Conversation",
                "default": True,
                "description": "Continue the session created by the previous message.",
            },
            "timeout": {
                "type": "number",
                "label": "Timeout (seconds)",
                "default": 600,
                "min": 10,
            },
            "max_turns": {
                "type": "number",
                "label": "Max Turns",
                "required": False,
                "description": "Optional agentic turn limit for print mode.",
            },
            "allowed_tools": {
                "type": "text",
                "label": "Allowed Tools",
                "required": False,
                "placeholder": "Read,Bash(git status *)",
                "description": (
                    "Comma-separated tools to auto-approve without prompting. "
                    "Uses Claude Code permission rule syntax."
                ),
            },
            "system_prompt": {
                "type": "textarea",
                "label": "System Prompt",
                "required": False,
                "description": "Replace the default system prompt when set.",
            },
            "append_system_prompt": {
                "type": "textarea",
                "label": "Append System Prompt",
                "required": False,
                "description": "Appended to the default system prompt when set.",
            },
            "bare": {
                "type": "checkbox",
                "label": "Bare Mode",
                "default": False,
                "description": (
                    "Skip hooks, plugins, auto-memory, and CLAUDE.md discovery. "
                    "Recommended for scripted CI-like runs; with bare mode, "
                    "API-key auth is preferred over OAuth."
                ),
            },
            "dangerously_skip_permissions": {
                "type": "checkbox",
                "label": "Skip All Permissions",
                "default": False,
                "description": (
                    "Equivalent to --dangerously-skip-permissions. Keep disabled "
                    "unless running in a trusted sandbox."
                ),
            },
        }

    @classmethod
    def request_schema(cls) -> dict:
        return {
            "prompt": {"type": "textarea", "label": "Prompt", "required": False},
            "session_id": {
                "type": "text",
                "label": "Session ID",
                "required": False,
            },
            "new_session": {
                "type": "checkbox",
                "label": "Start New Session",
                "default": False,
            },
            "action": {
                "type": "select",
                "label": "Action",
                "options": ["status", "send", "reset_session"],
                "default": "status",
            },
        }

    @classmethod
    def output_schema(cls) -> dict:
        return {
            "success": {"type": "bool", "label": "Success"},
            "status": {"type": "text", "label": "Status"},
            "response": {"type": "textarea", "label": "Response"},
            "session_id": {"type": "text", "label": "Session ID"},
            "model": {"type": "text", "label": "Model"},
            "error": {"type": "text", "label": "Error"},
            "events": {"type": "*", "label": "Claude Events"},
            "usage": {"type": "*", "label": "Usage"},
        }

    @classmethod
    def icon(cls) -> str:
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        try:
            with open(icon_path, "r", encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return ""

    def supports_chat(self) -> bool:
        return True

    def run(self, request: dict) -> dict:
        request = request or {}
        action = str(request.get("action") or "status")
        prompt = request.get("prompt") or request.get("message")

        if action == "reset_session":
            self.reset_session()
            return self._status_result("ready")
        if action == "send" or prompt:
            payload = dict(request)
            payload["content"] = prompt or ""
            return self.handle_message(payload, files=request.get("files"))
        return self.get_status()

    def get_status(self) -> dict:
        """Return installation and authentication health without changing auth state."""
        claude_path = self._find_claude_cli()
        if not claude_path:
            return self._status_result(
                "not_installed",
                error="Claude CLI was not found. Install it or configure claude_path.",
            )

        version = self._command_text([claude_path, "--version"], timeout=10)
        if self.auth_method == self.AUTH_API_KEY:
            if not self.api_key:
                return self._status_result(
                    "authentication_required",
                    error="Authentication method is api_key, but no API key is configured.",
                    version=version,
                )
            return self._status_result("ready", version=version)

        try:
            completed = subprocess.run(
                [claude_path, "auth", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
                env=self._build_environment(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return self._status_result("error", error=str(exc), version=version)

        login_output = self._combined_output(completed).strip()
        if completed.returncode == 0 and self._auth_status_logged_in(login_output):
            return self._status_result("ready", version=version)
        return self._status_result(
            "authentication_required",
            error=login_output or "Claude CLI login is required.",
            version=version,
        )

    def authentication_instructions(self) -> dict:
        """Return safe login instructions; never launch hidden interactive login."""
        claude_path = self._find_claude_cli() or "claude"
        return {
            "status": "authentication_required",
            "login_command": f"{claude_path} auth login",
            "message": (
                "Run login command in a visible terminal. Claude CLI owns and stores "
                "credentials; this datasource does not receive them."
            ),
        }

    def reset_session(self) -> None:
        with self._lock:
            self.session_id = None

    def handle_message(self, message: Any, files: list | None = None) -> dict:
        """Send one turn and return normalized response plus Claude CLI events."""
        with self._lock:
            normalized = self._normalize_message(message, files)
            prompt = normalized["prompt"].strip()
            attachments = normalized["files"]
            requested_session_id = normalized["session_id"]
            new_session = normalized["new_session"]
            requested_working_directory = normalized["working_directory"]
            requested_permission_mode = normalized["permission_mode"]

            if not prompt and not attachments:
                return self._error_result("invalid_request", "Message is empty.")

            status = self.get_status()
            if status["status"] != "ready":
                result = self._error_result(status["status"], status["error"])
                result["authentication"] = self.authentication_instructions()
                return result

            if new_session:
                self.session_id = None
            if requested_session_id:
                self.session_id = requested_session_id
            if requested_working_directory:
                workspace = os.path.abspath(os.path.expanduser(requested_working_directory))
                if not os.path.isdir(workspace):
                    return self._error_result(
                        "invalid_request", f"Working directory does not exist: {workspace}"
                    )
                self.working_directory = workspace
            if requested_permission_mode:
                if requested_permission_mode not in self.PERMISSION_MODES:
                    return self._error_result(
                        "invalid_request",
                        f"Unsupported permission mode: {requested_permission_mode}",
                    )
                self.permission_mode = requested_permission_mode

            with self._prepared_attachments(attachments) as prepared:
                effective_prompt = prompt
                if prepared["text_blocks"]:
                    effective_prompt = self._append_text_attachments(
                        effective_prompt, prepared["text_blocks"]
                    )
                if prepared["file_paths"]:
                    effective_prompt = self._append_file_path_hints(
                        effective_prompt, prepared["file_paths"]
                    )
                if not effective_prompt:
                    effective_prompt = "Review the attached file."

                resume_id = self.session_id if self.resume_session else None
                command = self._build_command(
                    effective_prompt,
                    session_id=resume_id,
                    extra_dirs=prepared["extra_dirs"],
                )
                BBLogger.log(
                    "[SubjectiveClaudeCliDataSource] Starting Claude turn "
                    f"mode={'resume' if resume_id else 'new'} "
                    f"session={resume_id or '-'} workspace={self.working_directory}"
                )
                result = self._execute_command(command)

            if result["success"] and result.get("session_id"):
                self.session_id = result["session_id"]
            elif not result.get("session_id"):
                result["session_id"] = self.session_id or ""
            return result

    def cancel(self) -> bool:
        """Stop the CLI child of an in-flight turn. Called from another thread.

        Returns True if a running process was signalled. Also latches, so a cancel that
        arrives in the gap between handle_message() and the spawn still takes effect
        instead of being silently lost.
        """
        with self._process_lock:
            self._cancelled = True
            process = self._active_process
        if process is None or process.poll() is not None:
            return False
        BBLogger.log("[SubjectiveClaudeCliDataSource] Cancelling Claude turn")
        self._signal_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._signal_group(process, signal.SIGKILL)
        return True

    @staticmethod
    def _signal_group(process: subprocess.Popen, sig: int) -> None:
        """Signal the whole process group.

        The CLI spawns its own children; signalling only the direct child leaves those
        orphaned, still running and still holding the session. The turn is started with
        start_new_session=True precisely so the group can be taken down as a unit.
        """
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill() if sig == signal.SIGKILL else process.terminate()
            except OSError:
                pass

    def _execute_command(self, command: list[str]) -> dict:
        try:
            with self._process_lock:
                if self._cancelled:
                    return self._error_result("cancelled", "Stopped before Claude started.")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=self._build_environment(),
                    cwd=self._valid_working_directory(),
                    # Own process group, so cancel() can take the CLI and everything it
                    # spawned down together.
                    start_new_session=True,
                )
                self._active_process = process
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self._signal_group(process, signal.SIGKILL)
                stdout, stderr = process.communicate()
                BBLogger.log(
                    f"[SubjectiveClaudeCliDataSource] Claude turn timed out after {self.timeout}s"
                )
                return self._error_result(
                    "timeout", f"Claude execution timed out after {self.timeout} seconds."
                )
            finally:
                with self._process_lock:
                    self._active_process = None
            if self._cancelled:
                return self._error_result("cancelled", "Stopped by the operator.")
            completed = subprocess.CompletedProcess(
                command, process.returncode, stdout or "", stderr or ""
            )
        except OSError as exc:
            BBLogger.log(f"[SubjectiveClaudeCliDataSource] Failed to start Claude: {exc}")
            return self._error_result("execution_error", str(exc))

        parsed = self._parse_cli_output(completed.stdout or "")
        if completed.returncode != 0:
            error_text = (
                completed.stderr or parsed.get("error") or parsed.get("raw_text") or ""
            ).strip()
            status = self._classify_error(error_text or (parsed.get("result") or ""))
            BBLogger.log(
                "[SubjectiveClaudeCliDataSource] Claude turn failed "
                f"exit={completed.returncode} status={status}: {error_text[:500]}"
            )
            result = self._error_result(
                status, error_text or f"Claude exited with code {completed.returncode}."
            )
            result.update(
                {
                    "events": parsed["events"],
                    "session_id": parsed["session_id"] or self.session_id or "",
                    "usage": parsed["usage"],
                    "exit_code": completed.returncode,
                    "model": parsed.get("model") or self.model,
                }
            )
            return result

        response = (parsed["assistant_message"] or "").strip()
        if not response:
            return {
                **self._error_result(
                    "invalid_response",
                    "Claude completed without an assistant message.",
                ),
                "events": parsed["events"],
                "session_id": parsed["session_id"] or self.session_id or "",
                "usage": parsed["usage"],
                "exit_code": completed.returncode,
            }

        BBLogger.log(
            "[SubjectiveClaudeCliDataSource] Claude turn completed "
            f"session={parsed['session_id'] or self.session_id or '-'}"
        )
        return {
            "success": True,
            "status": "completed",
            "response": response,
            "session_id": parsed["session_id"] or self.session_id or "",
            "model": parsed.get("model") or self.model,
            "error": "",
            "events": parsed["events"],
            "usage": parsed["usage"],
            "exit_code": completed.returncode,
        }

    def _build_command(
        self,
        message: str,
        *,
        session_id: str | None = None,
        extra_dirs: list[str] | None = None,
    ) -> list[str]:
        claude_path = self._find_claude_cli()
        if not claude_path:
            raise RuntimeError("Claude CLI not found")

        command = [claude_path, "-p", "--output-format", "json"]

        if self.bare:
            command.append("--bare")
        if session_id:
            command.extend(["--resume", session_id])
        if self.model:
            command.extend(["--model", self.model])
        if self.permission_mode and self.permission_mode != "default":
            command.extend(["--permission-mode", self.permission_mode])
        if self.dangerously_skip_permissions:
            command.append("--dangerously-skip-permissions")
        if self.max_turns is not None:
            command.extend(["--max-turns", str(self.max_turns)])
        if self.allowed_tools:
            tools = [
                part.strip()
                for part in self.allowed_tools.replace("\n", ",").split(",")
                if part.strip()
            ]
            if tools:
                command.append("--allowedTools")
                command.extend(tools)
        if self.system_prompt:
            command.extend(["--system-prompt", self.system_prompt])
        elif self.append_system_prompt:
            command.extend(["--append-system-prompt", self.append_system_prompt])

        for directory in extra_dirs or []:
            command.extend(["--add-dir", directory])

        # Variadic options (--add-dir, --allowedTools) otherwise consume the prompt.
        # The terminator also preserves prompts beginning with a dash.
        command.extend(["--", message])
        return command

    def _parse_cli_output(self, output: str) -> dict:
        """Parse Claude CLI print-mode output.

        Prefer a single JSON object (``--output-format json``). Also accept
        newline-delimited stream-json for compatibility.
        """
        events: list[dict[str, Any]] = []
        assistant_message = ""
        session_id = ""
        usage: dict[str, Any] = {}
        model = ""
        error = ""
        raw_lines: list[str] = []

        stripped = (output or "").strip()
        if not stripped:
            return {
                "events": [],
                "assistant_message": "",
                "session_id": "",
                "usage": {},
                "model": "",
                "error": "",
                "raw_text": "",
            }

        # Single-object JSON (preferred with --output-format json).
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return self._extract_from_payload(payload, events_seed=[payload])
        except json.JSONDecodeError:
            pass

        # NDJSON / stream-json fallback.
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raw_lines.append(line)
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)

            event_type = str(event.get("type") or "")
            if event.get("session_id") and not session_id:
                session_id = str(event.get("session_id") or "")
            if event.get("model") and not model:
                model = str(event.get("model") or "")

            if event_type == "result" or event.get("subtype") in {
                "success",
                "error",
            }:
                extracted = self._extract_from_payload(event, events_seed=[])
                if extracted["assistant_message"]:
                    assistant_message = extracted["assistant_message"]
                if extracted["session_id"]:
                    session_id = extracted["session_id"]
                if extracted["usage"]:
                    usage = extracted["usage"]
                if extracted["model"]:
                    model = extracted["model"]
                if extracted["error"]:
                    error = extracted["error"]

            if event_type in {"assistant", "message"} and not assistant_message:
                text = self._extract_text(event)
                if text:
                    assistant_message = text

        if not assistant_message and raw_lines:
            assistant_message = "\n".join(raw_lines)

        return {
            "events": events,
            "assistant_message": assistant_message,
            "session_id": session_id,
            "usage": usage,
            "model": model,
            "error": error,
            "raw_text": "\n".join(raw_lines),
        }

    def _extract_from_payload(
        self, payload: dict[str, Any], *, events_seed: list[dict[str, Any]]
    ) -> dict:
        session_id = str(
            payload.get("session_id")
            or payload.get("sessionId")
            or payload.get("thread_id")
            or ""
        )
        model = str(payload.get("model") or "")
        usage = self._extract_usage(payload)
        error = ""
        if payload.get("is_error") or payload.get("subtype") == "error":
            error = str(
                payload.get("error")
                or payload.get("result")
                or payload.get("message")
                or "Claude reported an error."
            )

        assistant_message = ""
        for key in ("result", "text", "output_text", "message", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                # Prefer the dedicated result field for print-mode JSON.
                if key == "result" or not assistant_message:
                    assistant_message = value
                if key == "result":
                    break
        if not assistant_message:
            assistant_message = self._extract_text(payload)

        return {
            "events": list(events_seed),
            "assistant_message": assistant_message,
            "session_id": session_id,
            "usage": usage,
            "model": model,
            "error": error,
            "raw_text": "",
        }

    @staticmethod
    def _extract_usage(payload: dict[str, Any]) -> dict[str, Any]:
        usage: dict[str, Any] = {}
        nested = payload.get("usage")
        if isinstance(nested, dict):
            usage.update(nested)
        for key in (
            "total_cost_usd",
            "duration_ms",
            "duration_api_ms",
            "num_turns",
            "input_tokens",
            "output_tokens",
        ):
            if key in payload and payload[key] is not None:
                usage[key] = payload[key]
        return usage

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        direct = payload.get("text") or payload.get("output_text") or payload.get("result")
        if isinstance(direct, str):
            return direct

        message = payload.get("message")
        if isinstance(message, dict):
            nested = SubjectiveClaudeCliDataSource._extract_text(message)
            if nested:
                return nested
        if isinstance(message, str):
            return message

        content = payload.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _auth_status_logged_in(output: str) -> bool:
        text = (output or "").strip()
        if not text:
            return False
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            lowered = text.lower()
            return "logged in" in lowered or '"loggedin": true' in lowered.replace(" ", "")
        if isinstance(payload, dict):
            if "loggedIn" in payload:
                return bool(payload.get("loggedIn"))
            if "logged_in" in payload:
                return bool(payload.get("logged_in"))
            # Some versions only emit authMethod/email when authenticated.
            return bool(payload.get("authMethod") or payload.get("email"))
        return False

    def _normalize_message(self, message: Any, files: list | None) -> dict:
        prompt = ""
        embedded_files: list[Any] = []
        session_id = ""
        new_session = False
        working_directory = ""
        permission_mode = ""

        if isinstance(message, dict):
            prompt_value = (
                message.get("content")
                if message.get("content") is not None
                else message.get("text", message.get("prompt", message.get("message", "")))
            )
            prompt = str(prompt_value or "")
            for key in ("files", "attachments"):
                value = message.get(key)
                if isinstance(value, list):
                    embedded_files.extend(value)
            session_id = str(message.get("session_id") or "")
            new_session = self._coerce_bool(message.get("new_session", False))
            working_directory = str(
                message.get("working_directory") or message.get("workspace") or ""
            ).strip()
            permission_mode = str(
                message.get("permission_mode") or message.get("sandbox_mode") or ""
            ).strip()
        else:
            prompt = str(message or "")

        # Deduplicate. run() puts the whole request in `message` *and* passes
        # request["files"] again as `files`, so every attachment arrived twice — the
        # "same image arrived twice" reports. A payload carrying both "files" and
        # "attachments" doubles them the same way. Dedupe on content rather than
        # identity: the two copies are equal dicts, not the same object.
        all_files = []
        seen: set[str] = set()
        for item in [*embedded_files, *(files if isinstance(files, list) else [])]:
            fingerprint = json.dumps(item, sort_keys=True, default=str) if isinstance(item, dict) else repr(item)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            all_files.append(item)
        return {
            "prompt": prompt,
            "files": all_files,
            "session_id": session_id or None,
            "new_session": new_session,
            "working_directory": working_directory,
            "permission_mode": permission_mode,
        }

    @contextmanager
    def _prepared_attachments(self, files: list[Any]) -> Iterator[dict[str, list]]:
        with tempfile.TemporaryDirectory(prefix="subjective-claude-cli-") as temp_dir:
            text_blocks: list[dict[str, str]] = []
            file_paths: list[str] = []
            extra_dirs: list[str] = []

            for index, item in enumerate(files):
                prepared = self._prepare_attachment(item, temp_dir, index)
                if not prepared:
                    continue
                if prepared["kind"] == "text":
                    text_blocks.append(
                        {"name": prepared["name"], "text": prepared["text"]}
                    )
                elif prepared["kind"] == "file":
                    file_paths.append(prepared["path"])
                    parent = os.path.dirname(prepared["path"])
                    if parent and parent not in extra_dirs:
                        extra_dirs.append(parent)

            # Always grant tool access to the temp attachment directory when used.
            if file_paths and temp_dir not in extra_dirs:
                extra_dirs.append(temp_dir)
            yield {
                "text_blocks": text_blocks,
                "file_paths": file_paths,
                "extra_dirs": extra_dirs,
            }

    def _prepare_attachment(
        self, item: Any, temp_dir: str, index: int
    ) -> dict[str, str] | None:
        if isinstance(item, str):
            path = os.path.abspath(os.path.expanduser(item))
            return self._prepare_path_attachment(path)
        if not isinstance(item, dict):
            return None

        source_path = item.get("path") or item.get("file_path")
        if source_path:
            return self._prepare_path_attachment(
                os.path.abspath(os.path.expanduser(str(source_path)))
            )

        name = Path(str(item.get("name") or item.get("filename") or f"file-{index}"))
        safe_name = name.name or f"file-{index}"
        mime_type = str(
            item.get("mime_type")
            or item.get("type")
            or mimetypes.guess_type(safe_name)[0]
            or "application/octet-stream"
        )
        raw_value = item.get("data_base64")
        if raw_value is None:
            raw_value = item.get("content")
        if raw_value is None and isinstance(item.get("text"), str):
            return {"kind": "text", "name": safe_name, "text": item["text"]}
        if raw_value is None:
            return None

        try:
            raw = base64.b64decode(str(raw_value), validate=True)
        except (binascii.Error, ValueError):
            raw = str(raw_value).encode("utf-8", errors="replace")
        if len(raw) > self.MAX_ATTACHMENT_BYTES:
            return {
                "kind": "text",
                "name": safe_name,
                "text": f"[Attachment omitted: exceeds {self.MAX_ATTACHMENT_BYTES} bytes]",
            }

        # Binary / image attachments are written to temp so Claude can Read them.
        if mime_type.startswith("image/") or not self._looks_like_text(raw, mime_type):
            path = os.path.join(temp_dir, f"{index}-{safe_name}")
            with open(path, "wb") as handle:
                handle.write(raw)
            return {"kind": "file", "name": safe_name, "path": path}

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = "[Binary attachment; textual content unavailable]"
        return {"kind": "text", "name": safe_name, "text": text}

    def _prepare_path_attachment(self, path: str) -> dict[str, str] | None:
        if not os.path.isfile(path):
            return {
                "kind": "text",
                "name": os.path.basename(path) or "attachment",
                "text": f"[Attachment path not found: {path}]",
            }
        name = os.path.basename(path)
        mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if mime_type.startswith("image/"):
            return {"kind": "file", "name": name, "path": path}
        if os.path.getsize(path) > self.MAX_ATTACHMENT_BYTES:
            return {
                "kind": "text",
                "name": name,
                "text": f"[Attachment omitted: exceeds {self.MAX_ATTACHMENT_BYTES} bytes]",
            }
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            return {"kind": "text", "name": name, "text": text}
        except (OSError, UnicodeDecodeError):
            return {"kind": "file", "name": name, "path": path}

    @staticmethod
    def _looks_like_text(raw: bytes, mime_type: str) -> bool:
        if mime_type.startswith("text/") or mime_type in {
            "application/json",
            "application/xml",
            "application/javascript",
            "application/x-yaml",
            "application/yaml",
        }:
            return True
        if b"\x00" in raw[:1024]:
            return False
        try:
            raw[:4096].decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    @staticmethod
    def _append_text_attachments(prompt: str, blocks: list[dict[str, str]]) -> str:
        sections = [prompt] if prompt else []
        for block in blocks:
            sections.append(
                f"<attachment name={json.dumps(block['name'])}>\n"
                f"{block['text']}\n"
                "</attachment>"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _append_file_path_hints(prompt: str, paths: list[str]) -> str:
        sections = [prompt] if prompt else []
        for path in paths:
            sections.append(
                f"<attachment_path path={json.dumps(path)}>\n"
                "Read this attached file from the local filesystem.\n"
                "</attachment_path>"
            )
        return "\n\n".join(sections)

    def _find_claude_cli(self) -> str | None:
        if self._claude_path:
            return self._claude_path

        candidates: list[str] = []
        if self.configured_claude_path:
            candidates.append(os.path.expanduser(self.configured_claude_path))
        path_candidate = shutil.which("claude")
        if path_candidate:
            candidates.append(path_candidate)
        candidates.extend(
            [
                os.path.expanduser("~/.local/bin/claude"),
                os.path.expanduser("~/bin/claude"),
                "/usr/local/bin/claude",
                os.path.expanduser("~/.npm-global/bin/claude"),
                os.path.expanduser("~/AppData/Local/Programs/claude/claude.exe"),
                os.path.expanduser("~/AppData/Roaming/npm/claude.cmd"),
            ]
        )

        for candidate in candidates:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                self._claude_path = os.path.abspath(candidate)
                return self._claude_path
        return None

    def _build_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        if self.auth_method == self.AUTH_API_KEY and self.api_key:
            environment["ANTHROPIC_API_KEY"] = self.api_key
        return environment

    def _valid_working_directory(self) -> str | None:
        if self.working_directory and os.path.isdir(self.working_directory):
            return self.working_directory
        return None

    @staticmethod
    def _combined_output(completed: subprocess.CompletedProcess) -> str:
        return "\n".join(
            part.strip()
            for part in (completed.stdout or "", completed.stderr or "")
            if part and part.strip()
        )

    def _command_text(self, command: list[str], timeout: int) -> str:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._build_environment(),
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return self._combined_output(completed).strip()

    @staticmethod
    def _classify_error(error_text: str) -> str:
        lowered = error_text.lower()
        if any(
            token in lowered
            for token in (
                "login",
                "not authenticated",
                "unauthorized",
                "authentication_failed",
                "not logged in",
                "please run /login",
                "auth status",
            )
        ):
            return "authentication_required"
        if any(
            token in lowered
            for token in (
                "usage limit",
                "rate limit",
                "quota",
                "too many requests",
                "billing",
                "credit",
            )
        ):
            return "quota_limited"
        if "session" in lowered and any(
            token in lowered for token in ("not found", "unknown", "does not exist", "invalid")
        ):
            return "session_not_found"
        return "execution_error"

    def _status_result(
        self, status: str, *, error: str = "", version: str = ""
    ) -> dict:
        return {
            "success": status == "ready",
            "status": status,
            "response": "",
            "session_id": self.session_id or "",
            "model": self.model,
            "error": error,
            "events": [],
            "usage": {},
            "provider": "claude_cli",
            "cli_path": self._find_claude_cli() or "",
            "cli_version": version,
        }

    def _error_result(self, status: str, error: str) -> dict:
        return {
            "success": False,
            "status": status,
            "response": "",
            "session_id": self.session_id or "",
            "model": self.model,
            "error": error,
            "events": [],
            "usage": {},
        }

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _coerce_int(value: Any, default: int, *, minimum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, parsed)

    @staticmethod
    def _coerce_optional_int(value: Any, *, minimum: int) -> int | None:
        if value is None or value == "":
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed < minimum:
            return None
        return parsed
