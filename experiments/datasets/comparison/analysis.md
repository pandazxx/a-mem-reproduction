# Comparison Dataset — Analysis Template

Use this document to capture hypotheses **before** running the evaluation, results **after** running it, and what the results actually mean.

---

## Hypotheses (write these *before* looking at results)

| Category | Hypothesis | Why this should hold |
|---|---|---|
| Single-hop (Q1-4) | Both systems ≥ 90% accuracy | Trivial retrieval; failure here indicates reproduction bug |
| Two-hop (Q5-7) | Both ≥ 70%, tie or slight HippoRAG advantage | Light multi-hop; HippoRAG's graph slightly more natural |
| Deep multi-hop (Q8-10) | HippoRAG ≥ 50%, A-Mem ≤ 30% | PPR follows chains; A-Mem's 1-hop link traversal misses depth |
| Implicit conceptual (Q11-13) | A-Mem ≥ 50%, HippoRAG ≤ 30% | LLM-determined links + tags capture conceptual relations |
| Information update (Q14-16) | A-Mem ≥ 60%, HippoRAG ≤ 40% | Memory evolution should update stale facts; HippoRAG holds both |
| Compositional aggregation (Q17-19) | HippoRAG ≥ 60%, A-Mem ≤ 50% | PPR aggregates across entities; top-k retrieval misses long tail |
| Absence (Q20-22) | Both ≥ 80% abstention rate | Both should recognise missing info; failure = hallucination |

**Aggregate hypothesis:** neither system wins overall; the wins are category-specific. If one system wins everywhere, suspect a reproduction issue.

---

## Results table (fill in after running)

| Q | Category | Expected winner | HippoRAG answer | A-Mem answer | HippoRAG correct? | A-Mem correct? | Actual winner |
|---|---|---|---|---|---|---|---|
| Q01 | single_hop | tie | | | | | |
| Q02 | single_hop | tie | | | | | |
| Q03 | single_hop | tie | | | | | |
| Q04 | single_hop | tie | | | | | |
| Q05 | two_hop | tie | | | | | |
| Q06 | two_hop | tie | | | | | |
| Q07 | two_hop | tie | | | | | |
| Q08 | deep_multi_hop | HippoRAG | | | | | |
| Q09 | deep_multi_hop | HippoRAG | | | | | |
| Q10 | deep_multi_hop | tie (trick) | | | | | |
| Q11 | implicit_conceptual | A-Mem | | | | | |
| Q12 | implicit_conceptual | A-Mem | | | | | |
| Q13 | implicit_conceptual | A-Mem | | | | | |
| Q14 | information_update | A-Mem | | | | | |
| Q15 | information_update | A-Mem | | | | | |
| Q16 | information_update | A-Mem | | | | | |
| Q17 | compositional_aggregation | HippoRAG | | | | | |
| Q18 | compositional_aggregation | HippoRAG | | | | | |
| Q19 | compositional_aggregation | HippoRAG | | | | | |
| Q20 | absence_abstention | tie | | | | | |
| Q21 | absence_abstention | tie | | | | | |
| Q22 | absence_abstention | tie | | | | | |

---

## Per-category summary (fill in)

| Category | HippoRAG correct/total | A-Mem correct/total | Hypothesis confirmed? |
|---|---|---|---|
| single_hop | /4 | /4 | |
| two_hop | /3 | /3 | |
| deep_multi_hop | /3 | /3 | |
| implicit_conceptual | /3 | /3 | |
| information_update | /3 | /3 | |
| compositional_aggregation | /3 | /3 | |
| absence_abstention | /3 | /3 | |
| **Total** | **/22** | **/22** | |

---

## Interpretation guide

### When the expected winner wins clearly

That's confirmation of the architectural hypothesis. Worth writing up the *mechanism* — e.g., "HippoRAG correctly traced the 3-hop chain on Q8 because PPR propagated probability mass from the Amazon node through David and Sam to Marcus."

### When the expected winner loses

That's the more interesting case. Possible explanations:

| Failure mode | Likely cause | What to check |
|---|---|---|
| HippoRAG loses on multi-hop | PPR didn't propagate far enough; damping factor too high | Try lowering damping; inspect intermediate PPR scores |
| A-Mem loses on update questions | Memory evolution didn't trigger; or the evolution prompt was conservative | Check whether memory evolution actually rewrote the old TechCorp memory after Week 7 |
| A-Mem loses on implicit conceptual | LLM-generated tags/context weren't rich enough | Inspect K, G, X fields of the relevant notes |
| HippoRAG loses on aggregation | Some relevant entities never got linked into the KG | Check OpenIE output — did entities get extracted? |
| Both hallucinate on absence questions | The QA reader LLM doesn't respect retrieval boundaries | Sharper system prompt; possibly need an explicit abstention mechanism |

### When both fail on the same question

The question is harder than expected, the dataset has a gap, or both systems have a shared blind spot. Examples to watch for:
- Temporal precedence (which fact came first / is more recent)
- Entity disambiguation across contexts
- Concept-level reasoning that neither's index naturally supports

---

## Findings narrative (fill in)

Write 3–5 paragraphs answering:

1. **Did the expected winners win in each category?**
2. **What was the most surprising result?** (a confirmed expectation is fine; a violated one is more interesting)
3. **What does this tell you about the architectural choices in each system?**
4. **Which findings would inform a reconsolidation-focused project design?**
5. **What's the single biggest gap that neither system addresses?**

---

## Per-question deep dive (optional)

For questions where the two systems disagreed, walk through the retrieval traces:

```
Question Q##:
HippoRAG retrieved: [memory IDs in order]
HippoRAG answer: ...
HippoRAG reasoning (if traceable): ...

A-Mem retrieved: [memory IDs in order]
A-Mem links followed: [m_x → m_y, ...]
A-Mem answer: ...
A-Mem note evolution log (relevant for update questions): ...

Why did they disagree?
What does this say about the architecture?
```

This is the highest-value analysis — these case studies are exactly what would become blog post material or paper anecdotes.

---

## Things to follow up on

After running this dataset, you'll likely have ideas for additional questions or memories that would test other distinctions. Capture them here for a v2:

- [ ] ...
- [ ] ...
- [ ] ...
