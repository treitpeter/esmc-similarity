# Validation on 26 September 2026

All 30 numerical/parser unit tests pass on Python 3.10 and 3.12. Tests cover
residue-only pooling, FASTA rejection, scale-stable cosine, zero/nonfinite inputs,
provenance mismatch, safe NPZ loading, exact eggNOG joins, missing rows, stable
ties, self-exclusion and output overwrite protection.

A separate **real-model** CPU smoke test used the committed three-sequence FASTA:

```bash
HF_HUB_OFFLINE=1 esmc-similarity embed examples/proteins.fasta toy.npz --threads 2
esmc-similarity search toy.npz toy.npz hits.tsv --top-k 2 --exclude-self \
  --query-annotations examples/toy.emapper.annotations \
  --reference-annotations examples/toy.emapper.annotations
```

Observed with Python 3.10.16, esm 3.1.3, PyTorch 2.6.0+cu124 running on **CPU**,
NumPy 1.26.4 and the checksum-verified checkpoint recorded in the source:

| Check | Result |
|---|---|
| Embedding shape | 3 × 960 |
| Identical sequence pair | cosine 1.0 |
| Original vs reversed toy sequence | cosine 0.980503494 |
| eggNOG fixture joins | 2 of 3 IDs on each side |
| Missing toy_b annotation | row_present=0, empty annotation fields |
| Self-ID exclusion | 6 rows, no identical query/reference IDs |

This smoke test first used an existing backend environment through a separate
local virtual environment. A second, **fresh isolated Python 3.10 environment**
was then installed using the documented CPU packages (torch 2.6.0+cpu,
torchvision 0.21.0+cpu) and `.[embed,test]`. Dependency checking passed; all 30
tests and the real-model embedding command passed there too. The weights were
identical between runs, and the maximum absolute embedding difference was 0.0.
Weights were
already cached, and Hugging Face offline mode was enabled during both inference
runs. No sequence upload or paid backend was involved. A first-time remote weight
download was not repeated; the pinned model revision was separately reachable.

**Interpretation:** identical sequences provide a basic deterministic control.
The reversed short toy sequence shares amino-acid composition and still scores
highly. Neither sequence has a validated function here, so this is evidence
against treating a large cosine as a universally calibrated functional claim,
not a benchmark of biological retrieval accuracy. Synthetic eggNOG fields test
file parsing only. CUDA, biological accuracy and database-scale throughput are
not validated by these checks.
