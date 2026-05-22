# A-Mem Reproduction

Reproduction of [A-Mem: Agentic Memory for LLM Agents (Xu et al., 2025)](https://arxiv.org/abs/2502.12110), a memory system that uses LLM-generated dynamic notes and inter-memory links instead of static vector embeddings.

This is a warmup / practice reproduction as part of a larger agent memory research project. The goal is **running, not matching published numbers exactly**.

## Status

🚧 Setting up. See [NOTES.md](NOTES.md) for ongoing learnings.

## Goals

- Get the A-Mem pipeline running end-to-end on a small slice (10–20 examples) of [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) or [LongMemEval](https://github.com/xiaowu0162/LongMemEval).
- Document the reproduction process honestly: what was easy, what was hard, what the codebase quality is like.
- Contrast the LLM-managed dynamic notes paradigm with the explicit knowledge graph approach (HippoRAG, reproduced separately).

## Original work

- **Paper:** https://arxiv.org/abs/2502.12110
- **Code:** https://github.com/agiresearch/A-mem

## Approach

This repo is **not a fork** of the original A-Mem. It pulls the original in as a dependency. The structure here adds:

- `docs/` — reading notes, design observations.
- `experiments/` — scripts to run reproduction experiments.
- `results/` — logged benchmark numbers.
- `NOTES.md` — running journal of what was easy, what was hard.

## Timeline

Weeks 4–5 of the broader research project timeline. Hard cap: 10 working days. If genuinely stuck, the work is abandoned and documented.

## License

MIT. See [LICENSE](LICENSE).
