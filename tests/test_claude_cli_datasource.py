import base64
import json
import os
import subprocess
import unittest
from unittest.mock import patch

from SubjectiveClaudeCliDataSource import SubjectiveClaudeCliDataSource


def completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def result_json(
    *,
    result="Done.",
    session_id="session-123",
    model="claude-sonnet",
    usage=None,
    is_error=False,
    total_cost_usd=0.01,
):
    payload = {
        "type": "result",
        "subtype": "error" if is_error else "success",
        "result": result,
        "session_id": session_id,
        "model": model,
        "is_error": is_error,
        "total_cost_usd": total_cost_usd,
        "duration_ms": 1200,
        "num_turns": 1,
    }
    if usage is not None:
        payload["usage"] = usage
    return json.dumps(payload) + "\n"


class SubjectiveClaudeCliDataSourceTests(unittest.TestCase):
    def datasource(self, **connection):
        defaults = {
            "auth_method": "existing_session",
            "working_directory": "/tmp",
            "permission_mode": "default",
        }
        defaults.update(connection)
        instance = SubjectiveClaudeCliDataSource(connection=defaults, config={})
        instance._claude_path = "/usr/bin/claude"
        return instance

    def ready(self, instance):
        instance.get_status = lambda: instance._status_result("ready")

    def test_v2_contract_and_safe_defaults(self):
        instance = self.datasource()
        self.assertEqual(instance.api_version(), "v2")
        self.assertTrue(instance.supports_chat())
        self.assertEqual(instance.auth_method, "existing_session")
        self.assertEqual(instance.permission_mode, "default")
        self.assertTrue(instance.resume_session)
        self.assertFalse(instance.dangerously_skip_permissions)
        self.assertFalse(instance.bare)
        self.assertIn("prompt", instance.request_schema())
        self.assertIn("session_id", instance.output_schema())
        self.assertIn("auth_method", instance.connection_schema())

    @patch("SubjectiveClaudeCliDataSource.subprocess.run")
    def test_status_reuses_existing_cli_login_without_starting_login(self, run):
        run.side_effect = [
            completed([], stdout="2.1.178 (Claude Code)\n"),
            completed(
                [],
                stdout=json.dumps(
                    {
                        "loggedIn": True,
                        "authMethod": "claude.ai",
                        "email": "user@example.com",
                    }
                )
                + "\n",
            ),
        ]
        result = self.datasource().get_status()

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "ready")
        self.assertIn("2.1.178", result["cli_version"])
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(["/usr/bin/claude", "auth", "status", "--json"], commands)
        self.assertFalse(any("login" in command for command in commands))

    @patch("SubjectiveClaudeCliDataSource.subprocess.run")
    def test_status_reports_login_required(self, run):
        run.side_effect = [
            completed([], stdout="2.1.178 (Claude Code)\n"),
            completed(
                [],
                returncode=1,
                stdout=json.dumps({"loggedIn": False}) + "\n",
            ),
        ]
        result = self.datasource().get_status()
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "authentication_required")

    def test_api_key_is_scoped_to_child_environment(self):
        original = os.environ.get("ANTHROPIC_API_KEY")
        instance = self.datasource(auth_method="api_key", api_key="test-secret")
        environment = instance._build_environment()

        self.assertEqual(environment["ANTHROPIC_API_KEY"], "test-secret")
        self.assertEqual(os.environ.get("ANTHROPIC_API_KEY"), original)

    def test_new_session_command_uses_print_json_layout(self):
        instance = self.datasource(
            model="sonnet",
            bare=True,
            max_turns=3,
            allowed_tools="Read,Bash(git status *)",
            append_system_prompt="Be concise.",
            permission_mode="acceptEdits",
        )
        command = instance._build_command("hello", extra_dirs=["/tmp/attachments"])

        self.assertEqual(command[:4], ["/usr/bin/claude", "-p", "--output-format", "json"])
        self.assertIn("--bare", command)
        self.assertIn("--model", command)
        self.assertIn("sonnet", command)
        self.assertIn("--permission-mode", command)
        self.assertIn("acceptEdits", command)
        self.assertIn("--max-turns", command)
        self.assertIn("3", command)
        self.assertIn("--allowedTools", command)
        self.assertIn("Read", command)
        self.assertIn("Bash(git status *)", command)
        self.assertIn("--append-system-prompt", command)
        self.assertIn("--add-dir", command)
        self.assertEqual(command[-1], "hello")
        self.assertNotIn("--resume", command)
        self.assertNotIn("--dangerously-skip-permissions", command)

    def test_resume_command_uses_native_session_id(self):
        instance = self.datasource(model="opus", dangerously_skip_permissions=True)
        command = instance._build_command("continue", session_id="session-abc")

        self.assertEqual(command[:4], ["/usr/bin/claude", "-p", "--output-format", "json"])
        self.assertIn("--resume", command)
        self.assertIn("session-abc", command)
        self.assertIn("--dangerously-skip-permissions", command)
        self.assertEqual(command[-1], "continue")

    def test_parser_supports_print_mode_json(self):
        output = result_json(
            result="Done.",
            session_id="session-123",
            usage={"input_tokens": 12, "output_tokens": 3},
        )
        parsed = self.datasource()._parse_cli_output(output)

        self.assertEqual(parsed["session_id"], "session-123")
        self.assertEqual(parsed["assistant_message"], "Done.")
        self.assertEqual(parsed["usage"]["output_tokens"], 3)
        self.assertEqual(parsed["usage"]["total_cost_usd"], 0.01)
        self.assertEqual(len(parsed["events"]), 1)

    def test_parser_supports_stream_json_result_line(self):
        lines = "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "session_id": "session-stream",
                        "model": "claude-sonnet",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "Streamed text"}]
                        },
                        "session_id": "session-stream",
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "Final answer",
                        "session_id": "session-stream",
                        "total_cost_usd": 0.02,
                    }
                ),
            ]
        )
        parsed = self.datasource()._parse_cli_output(lines)
        self.assertEqual(parsed["session_id"], "session-stream")
        self.assertEqual(parsed["assistant_message"], "Final answer")
        self.assertEqual(parsed["usage"]["total_cost_usd"], 0.02)
        self.assertEqual(len(parsed["events"]), 3)

    @patch("SubjectiveClaudeCliDataSource.subprocess.run")
    def test_second_message_resumes_session_created_by_first(self, run):
        instance = self.datasource()
        self.ready(instance)
        run.side_effect = [
            completed([], stdout=result_json(result="First", session_id="session-123")),
            completed([], stdout=result_json(result="Second", session_id="session-123")),
        ]

        first = instance.handle_message("hello")
        second = instance.handle_message("continue")

        self.assertTrue(first["success"])
        self.assertEqual(first["session_id"], "session-123")
        self.assertEqual(second["response"], "Second")
        second_command = run.call_args_list[1].args[0]
        self.assertIn("--resume", second_command)
        self.assertIn("session-123", second_command)

    @patch("SubjectiveClaudeCliDataSource.subprocess.run")
    def test_explicit_new_session_does_not_resume(self, run):
        instance = self.datasource(session_id="old-session")
        self.ready(instance)
        run.return_value = completed(
            [], stdout=result_json(result="New", session_id="new-session")
        )

        result = instance.handle_message({"content": "start", "new_session": True})

        self.assertEqual(result["session_id"], "new-session")
        command = run.call_args.args[0]
        self.assertNotIn("--resume", command)
        self.assertNotIn("old-session", command)

    @patch("SubjectiveClaudeCliDataSource.subprocess.run")
    def test_cli_error_is_structured_and_keeps_session(self, run):
        instance = self.datasource(session_id="session-123")
        self.ready(instance)
        run.return_value = completed([], returncode=1, stderr="session not found")

        result = instance.handle_message("continue")

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "session_not_found")
        self.assertEqual(result["session_id"], "session-123")
        self.assertEqual(result["exit_code"], 1)

    @patch("SubjectiveClaudeCliDataSource.subprocess.run")
    def test_timeout_is_structured(self, run):
        instance = self.datasource(timeout=10)
        self.ready(instance)
        run.side_effect = subprocess.TimeoutExpired(["claude"], timeout=10)

        result = instance.handle_message("hello")

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "timeout")

    def test_text_and_image_attachments_are_normalized(self):
        instance = self.datasource()
        files = [
            {
                "filename": "notes.txt",
                "content": base64.b64encode(b"important context").decode("ascii"),
            },
            {
                "filename": "pixel.png",
                "mime_type": "image/png",
                "content": base64.b64encode(b"not-a-real-png").decode("ascii"),
            },
        ]

        with instance._prepared_attachments(files) as prepared:
            self.assertEqual(prepared["text_blocks"][0]["text"], "important context")
            self.assertEqual(len(prepared["file_paths"]), 1)
            self.assertTrue(os.path.isfile(prepared["file_paths"][0]))
            self.assertTrue(prepared["extra_dirs"])

    @patch("SubjectiveClaudeCliDataSource.subprocess.run")
    def test_request_can_select_workspace_and_permission_mode(self, run):
        instance = self.datasource(
            working_directory="/tmp", permission_mode="default"
        )
        self.ready(instance)
        run.return_value = completed(
            [],
            stdout=result_json(result="Ready", session_id="session-workspace"),
        )
        result = instance.handle_message(
            {
                "content": "start",
                "workspace": "/tmp",
                "permission_mode": "plan",
                "new_session": True,
            }
        )
        self.assertTrue(result["success"])
        command = run.call_args.args[0]
        self.assertIn("--permission-mode", command)
        self.assertIn("plan", command)
        # Working directory is applied via subprocess cwd, not a CLI flag.
        self.assertEqual(run.call_args.kwargs.get("cwd"), "/tmp")

    def test_authentication_instructions_never_hide_login(self):
        instructions = self.datasource().authentication_instructions()
        self.assertIn("auth login", instructions["login_command"])
        self.assertIn("visible terminal", instructions["message"])


@unittest.skipUnless(
    os.environ.get("RUN_CLAUDE_CLI_LIVE_TEST") == "1",
    "set RUN_CLAUDE_CLI_LIVE_TEST=1 to use local Claude account",
)
class SubjectiveClaudeCliLiveTests(unittest.TestCase):
    def test_authenticated_print_turn(self):
        instance = SubjectiveClaudeCliDataSource(
            connection={
                "auth_method": "existing_session",
                "working_directory": os.getcwd(),
                "permission_mode": "default",
                "timeout": 120,
                "max_turns": 1,
            },
            config={},
        )
        result = instance.handle_message(
            "Reply with exactly: claude cli datasource works"
        )
        self.assertTrue(result["success"], result)
        self.assertIn("claude cli datasource works", result["response"].lower())
        self.assertTrue(result["session_id"])


if __name__ == "__main__":
    unittest.main()
