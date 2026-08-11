# Subjective Claude CLI Datasource

Resumable Subjective v2 chat datasource backed by local Anthropic Claude Code CLI.

## Authentication

Default `existing_session` mode reuses credentials already managed by Claude CLI:

```bash
claude auth status
claude auth login
```

Login is never started invisibly by the datasource. Device/browser login must remain visible to the user. API-key mode is also supported; the key is passed only to the child Claude process through `ANTHROPIC_API_KEY` and is not written into the global process environment.

## Conversation behavior

First message runs:

```bash
claude -p --output-format json "prompt"
```

Datasource captures `session_id` from JSON output. Following messages run:

```bash
claude -p --output-format json --resume SESSION_ID "next prompt"
```

Returned object contains:

```json
{
  "success": true,
  "status": "completed",
  "response": "assistant response",
  "session_id": "claude-session-uuid",
  "model": "",
  "error": "",
  "events": [],
  "usage": {}
}
```

Send `{ "content": "...", "new_session": true }` to start a fresh conversation. Send `session_id` to resume an explicit Claude session.

Text attachments are embedded into the prompt. Binary/image attachments are written to temporary files, exposed via `--add-dir`, and referenced in the prompt so Claude can read them. Temporary files are removed after the turn.

Optional connection controls map to Claude Code flags:

| Connection field | CLI flag |
| --- | --- |
| `model` | `--model` |
| `permission_mode` | `--permission-mode` |
| `max_turns` | `--max-turns` |
| `allowed_tools` | `--allowedTools` |
| `system_prompt` | `--system-prompt` |
| `append_system_prompt` | `--append-system-prompt` |
| `bare` | `--bare` |
| `dangerously_skip_permissions` | `--dangerously-skip-permissions` |
| `working_directory` | subprocess `cwd` |

## Status

`run({"action": "status"})` or `get_status()` returns one of:

- `ready`
- `not_installed`
- `authentication_required`
- `quota_limited`
- `execution_error`

## Tests

Set local dependency paths when running directly from a source checkout:

```bash
export PYTHONPATH=/subjective/libs/dependencies/subjective-abstract-data-source-package:/subjective/libs/dependencies/brainboost_data_source_logger_package:/subjective/libs/dependencies/brainboost_configuration_package:$PWD
python -m pytest -q tests/test_claude_cli_datasource.py
```

Live test consumes account capacity and is disabled by default:

```bash
RUN_CLAUDE_CLI_LIVE_TEST=1 python -m pytest -q tests/test_claude_cli_datasource.py
```
