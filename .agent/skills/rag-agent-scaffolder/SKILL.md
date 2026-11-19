---
name: rag-agent-scaffolder
description: Build local RAG + tool-calling layers and enforce 70/30 replay mixing for Phase 4 domain specialization.
---
# Instructions
1. For Phase 4 domain training, enforce a 70% Domain Data + 30% Phase-2 Generalist Replay mix to prevent catastrophic forgetting.
2. Keep fast-changing facts in a lightweight local Vector DB (ChromaDB/Qdrant) runnable on 6 GB VRAM / 16 GB RAM instead of baking them into weights.
3. Attach clean Python execution and search tools for agentic accuracy.
