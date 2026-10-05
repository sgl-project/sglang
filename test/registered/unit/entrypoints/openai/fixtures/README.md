`beam_chat_template.jinja` is an unmodified snapshot of Olympus's
`olympus/inference/libs/wire_format/grammars/harmony_xml/chat_template.jinja`
at commit `a93f2d6f5da5a33dd22fc79a2dd3090824abbaa2`.

Responses tests render this template to cover native built-in placement and
`refl.*` call/result routing. Tokenization and generation remain test doubles;
no model checkpoint or external service is required.

`beam_response_template.json` contains the `RESPONSE_TEMPLATE` metadata from
`response_template.py` in the same directory and commit. Tests cover both this
checkpoint profile and SGLang's fallback, including normalization of `refl.*`
wire recipients to bare tool names.

`beam_bash_tool.json` contains the name, description and parameter schema from
Olympus's `olympus/projects/minos/environments/mini_swe_agent/bash_tool_spec.py`
at the same commit. The shell tests compare both the full schema and the exact
rendered system-tool declaration against this canonical training definition.
