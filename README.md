# ai-agent-council

A minimal Python implementation of the Braintrust multi-agent council pattern: Belbin-roled
agents running a **divergent → critique → synthesis → finishing → orchestrate** loop, where
cognitive diversity comes from using *different model families* — not different prompts on the
same model.

## Install

```bash
uv sync
```

## Run

```bash
# Scaffold a council config from a shipped template
uv run council init council.yaml --template workstation-4agent

# Validate it
uv run council validate council.yaml

# Run the council on a task
uv run council run "Write a haiku about recursion." \
  --config council.yaml --transcript /tmp/run1.json

# Stream tokens live to stderr as each agent writes
uv run council run "..." --config council.yaml --stream
```

## Configurations

Three hardware-tier templates ship inside the package
(`src/ai_agent_council/templates/`) — `council init` copies one out:

| Template | Size | Hardware | Notes |
|---|---|---|---|
| `laptop-4agent` | 4 agents | laptop / 8GB VRAM or CPU | all small local models |
| `workstation-4agent` | 4 agents | 12GB VRAM (e.g. RTX 4070) | local + cloud orchestrator |
| `server-6agent` | 6 agents | 24GB+ VRAM / multi-GPU | full Belbin roster |

## Design principles (load-bearing)

These are invariants, not conventions:

1. **Size cap 4–8.** Enforced by pydantic validators.
2. **Cognitive diversity.** No two agents may share a model family (`gemma`, `deepseek`, `phi`,
   `qwen`, `claude`, `gpt`, …). Violations raise a `ValidationError` at config load.
3. **Anti-anchoring.** In the divergent phase, each agent's prompt is built from the task
   alone — it *physically cannot* see peer drafts. Enforced by the function signature of
   `render_divergent_prompt(task: str) -> str`, plus a regression test.
4. **Pixar rule in critique.** Critics diagnose, they do not prescribe.
5. **Explicit permission to disagree.** Baked into every system prompt.
6. **Temperature varies by role.** Ideator 0.9, Critic 0.2, Finisher 0.1.
7. **Thin Orchestrator.** Routes and synthesizes; does not produce substantive content.

## Features

- **Cost + token tracking.** Every run reports `total_cost_usd` and `total_tokens` on the
  `CouncilResult`; the CLI prints a one-line summary after each run. Local Ollama models
  resolve to $0.00 because LiteLLM doesn't price them.
- **Token streaming.** Pass `tokens=callback` to `Council.run()` or use `--stream` on the CLI
  to get live per-token output as each agent composes its response.
- **Tool calling for the Finisher.** Built-in safe tools are in `ai_agent_council.tools`:
  `current_time`, `calculate` (AST-validated arithmetic), `read_file` (sandboxed to the
  working directory). Enable per-agent by listing them in the YAML's `tools:` field.
  Register your own via `ai_agent_council.tools.register(Tool(...))`.
- **Retrospective loop.** Set `retrospective: true` on a config and every run appends 1-3
  Critic-extracted lessons to `~/.local/share/ai_agent_council/retrospectives/<name>.jsonl`.
  On the next run of the same council name, the most-recent lessons are re-injected into
  every agent's system prompt as *"Recent lessons from prior runs"* — closing the feedback
  loop the design brief calls for.

## Library API

```python
from ai_agent_council import Council

council = Council.from_yaml("council.yaml")
result = await council.run(
    "Write a haiku about recursion.",
    tokens=lambda agent, chunk: print(chunk, end="", flush=True),
)
print(result.final_answer)
print(f"Cost: ${result.total_cost_usd:.4f}")
```
