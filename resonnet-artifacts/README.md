# Embedded Artifacts

Embedded Artifacts Experiement with Claude 4.

#### Interface relegates valuable outputs (reports, rendered code, rich content) to side-window.

![Manus-Screenshots](https://github.com/user-attachments/assets/60b023a0-f6b5-4679-84bb-b908a6694b8e)

#### Replace the side-window approach with inline artifacts embedded directly within responses.
- Visualisations appear contextually within the conversation flow
- Interactive elements are immediately accessible without switching focus
- Artifacts can be expanded to fullscreen


#### Expanding Visualisations
ChatGPT recently added RDKit support. Ideally, we would enable the LLM to leverage any rendering library or visualisation tool that best fits the task.

![ChatGPT RDKit](https://github.com/user-attachments/assets/bfe5128a-ef2a-4113-8965-68e209c7f4aa)

#### System Prompt Guidance
- All artifacts render directly within the conversation flow, not in separate windows
- Example artifact types
  - Interactive simulations with parameter controls
  - Scientific/Molecular Visualisation
  - Three.js
  - Auto-notebook (Educational notebooks with executable code)

## Reading List

- [Reconstructed Artifact System Prompt for Claude 4](https://simonwillison.net/2025/May/25/claude-4-system-prompt/#artifacts-the-missing-manual)
- [Supposed Claude Sonnet 3.5 Artifact System Prompt](https://gist.github.com/dedlim/6bf6d81f77c19e20cd40594aa09e3ecd)
- [Manus AI Tools and Prompts](https://gist.github.com/jlia0/db0a9695b3ca7609c9b1a08dcbf872c9)
- [Sonnet 4 System Prompt w/ Artifact Guidance](https://github.com/asgeirtj/system_prompts_leaks/blob/main/Anthropic/claude-sonnet-4.md)
