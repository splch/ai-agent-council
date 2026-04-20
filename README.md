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
# Validate a config
uv run council validate configs/workstation-4agent.yaml

# Run the council on a task
uv run council run "Write a haiku about recursion." \
  --config configs/workstation-4agent.yaml \
  --transcript /tmp/run1.json
```

## Configurations

Three hardware tiers ship out of the box — pick the one that matches your hardware:

| Config | Size | Hardware | Notes |
|---|---|---|---|
| `configs/laptop-4agent.yaml` | 4 agents | laptop / 8GB VRAM or CPU | all small local models |
| `configs/workstation-4agent.yaml` | 4 agents | 12GB VRAM (e.g. RTX 4070) | local + cloud orchestrator |
| `configs/server-6agent.yaml` | 6 agents | 24GB+ VRAM / multi-GPU | full Belbin roster |

Scaffold your own:

```bash
uv run council init my-council.yaml --template workstation-4agent
```

## Design principles (load-bearing)

These are invariants, not conventions:

1. **Size cap 4–8.** Enforced by pydantic validators.
2. **Cognitive diversity.** No two agents may share a model family (`gemma`, `deepseek`, `phi`,
   `qwen`, `claude`, `gpt`, …). Violations raise a `ValidationError` at config load.
3. **Anti-anchoring.** In the divergent phase, each agent's prompt is built from the task
   alone — it *physically cannot* see peer drafts. Enforced by the function signature of
   `_render_divergent_prompt(task: str) -> str`, plus a regression test.
4. **Pixar rule in critique.** Critics diagnose, they do not prescribe.
5. **Explicit permission to disagree.** Baked into every system prompt.
6. **Temperature varies by role.** Ideator 0.9, Critic 0.2, Finisher 0.1.
7. **Thin Orchestrator.** Routes and synthesizes; does not produce substantive content.

## Library API

```python
from ai_agent_council import Council

council = Council.from_yaml("configs/workstation-4agent.yaml")
result = await council.run("Write a haiku about recursion.")
print(result.final_answer)
```
