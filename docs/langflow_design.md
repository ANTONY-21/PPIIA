# PPIIA as a Langflow workflow

The production pipeline in `ppiia_ai.py` is plain Python (easier to test, version
and deploy), but it is deliberately shaped like an LLM-chain graph, so the same
design can be reproduced in **Langflow** for the workflow-design part of the
project. This document maps the code to Langflow components.

## Graph

```
┌────────────┐   ┌──────────────┐   ┌─────────────────┐
│ File/URL   │──▶│ Text Cleaner │──▶│  Prompt: Stage 1 │──▶ Anthropic ──▶ Parse JSON ─┐
│ Loader     │   │ (prepare)    │   │  classify+meta   │    (Claude)                  │
└────────────┘   └──────┬───────┘   └─────────────────┘                               │
                        │           ┌─────────────────┐                               ▼
                        ├──────────▶│  Prompt: Stage 2 │──▶ Anthropic ──▶ Parse ──▶ ┌──────────┐
                        │           │  chronology      │                            │  Merge / │
                        ├──────────▶│  Prompt: Stage 3 │──▶ Anthropic ──▶ Parse ──▶ │ Combine  │──▶ JSON out
                        │           │  impact + risk   │                            │  (dict)  │
                        └──────────▶│  Prompt: Stage 4 │──▶ Anthropic ──▶ Parse ──▶ └──────────┘
                                    │  citizen summary │
                                    └─────────────────┘
```

## Component mapping

| Langflow component | Equivalent in this repo |
|---|---|
| **File Loader / URL Loader** | `ppiia.extract_text_from_pdf`, `ppiia.extract_text_from_url` |
| **Text processing node** | `ppiia_ai.prepare_text` (keeps clause structure, strips page furniture) |
| **Prompt template ×4** | `_metadata_instruction`, `_chronology_instruction`, `_impact_instruction`, `_summary_instruction` + the shared `SYSTEM_PROMPT` |
| **Anthropic model node ×4** | `ClaudeEngine.run_stage` (model `claude-opus-5`, structured outputs) |
| **Structured output parser** | The pydantic schemas in `schemas.py` (`LLMMetadata`, `LLMChronology`, `LLMImpact`, `LLMSummary`) |
| **Combine / Merge data node** | `ppiia_ai._assemble` → `BillAnalysis` |
| **Chat/JSON output** | `BillAnalysis.model_dump_json()` |

## Rebuilding it in Langflow

1. `pip install langflow` and start with `langflow run`.
2. Drag a **File** component (bill PDF) into a new flow; wire it to a **Parser** →
   **Prompt** component containing Stage 1's instruction text (copy it from
   `ppiia_ai.py`), with the bill text as a `{bill}` variable.
3. Wire the prompt into an **Anthropic** model component (model `claude-opus-5`,
   your `ANTHROPIC_API_KEY`); attach a **Structured Output** component whose JSON
   schema mirrors `LLMMetadata` (Langflow's structured-output component accepts a
   JSON schema — generate it with `LLMMetadata.model_json_schema()`).
4. Repeat for stages 2–4 (they can run in parallel branches off the same parsed
   text), then join the four outputs with a **Combine Data** component into one
   JSON record.
5. Export the flow as JSON (`⚙ → Export`) to keep it in the repo.

Two practical notes if you build the flow:

- Stage 2–4 prompts in the code interpolate the bill *title* found by stage 1.
  In Langflow, either chain stage 1 before the others (sequential, like the
  code) or drop the title variable and run all four in parallel.
- The code sends the bill text with a `cache_control` block so Anthropic's
  prompt cache pays for the repeated text only once; Langflow's stock Anthropic
  component doesn't expose cache control, which is one reason the production
  path stays in Python.
