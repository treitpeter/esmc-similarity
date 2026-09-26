# Decisions and empirical questions

1. **Local 300M only.** Meets the no-paid-model requirement and keeps compute lower
   than larger ESM-C variants. We have not demonstrated that 300M is biologically
   optimal. Compare retrieval accuracy, latency and memory on a held-out benchmark
   before arguing that any model size is best.
2. **Residue-only mean pooling.** The SDK returns two extra positions (BOS/EOS).
   Averaging them gives special tokens a fraction 2/(L+2) of the mean, a length-
   dependent contribution. The unit fixture verifies that changing those tokens
   does not change the pooled residue vector. Mean pooling is simple; it can dilute
   short functional domains. It is a baseline, not a proven optimal representation.
3. **No silent truncation.** A C-terminal domain could otherwise disappear. Rejecting
   an overlength sequence makes lost coverage visible. The parser regression proves
   rejection, not the scientific accuracy of long-sequence embeddings.
4. **Cosine with explicit provenance.** Unit-vector dot products ignore magnitude.
   Analytic fixtures cover identical, orthogonal and opposite vectors, extreme
   scaling, ties, missing data and incompatible representations. These are numerical
   correctness checks, not evidence of function prediction.
5. **eggNOG and similarity side by side.** Preserve direct query annotations and
   reference annotations separately, along with seed/evalue/score when supplied.
   A close embedding neighbour cannot turn an unannotated query into a confirmed
   function. The parser checks exact ID joins and missing rows.

Before a biological claim: use a reference set with defensible labels; split by
sequence-family clusters to reduce homology leakage; hold out the test labels;
compare eggNOG/sequence-search alone, cosine alone and a prespecified combination.
Report per-class precision/recall, coverage, abstention, class imbalance, confidence
intervals, runtime and memory. Investigate discordant and unannotated proteins.
Calibrate any cosine threshold on held-out data rather than inventing a universal
cutoff. Separate curated experimental labels from transferred annotations.

A COG class and a task-specific biological role need not describe the same thing.
A contingency table between them is association, not automatic agreement or
validation. Keep missing annotations distinct from low sequence identity and from
evidence that a protein has no function.

Graph view: k-nearest-neighbour edges are directed and depend on sampling density.
Mutual-neighbour filtering and thresholds change degrees, components and retained
coverage. Compare those quantities before presenting clusters as biological groups.
