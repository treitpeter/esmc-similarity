# ESM-C similarity

**Built with ESM.** A small local tool: protein FASTA → ESM-C 300M embeddings →
cosine neighbours, with optional eggNOG-mapper annotations alongside each match.

No paid inference service or API key. Inference runs on your CPU or GPU; the first
run downloads approximately 1.3 GB of model weights. Your sequences stay local.
Compute and storage are your own. This generates **embeddings**, not new proteins.

## Use

Python 3.10 or 3.11 is recommended for the pinned embedding backend.

```bash
python -m venv .venv
source .venv/bin/activate
# Optional CPU-only PyTorch installation (avoids downloading CUDA libraries):
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install '.[embed]'
esmc-similarity embed queries.fasta queries.npz
esmc-similarity embed reference.fasta reference.npz
esmc-similarity search queries.npz reference.npz hits.tsv --top-k 5 \
  --query-annotations queries.emapper.annotations \
  --reference-annotations reference.emapper.annotations
```

Omit either annotation option if you do not have that file. This tool **reads
existing eggNOG-mapper outputs**; it does not run eggNOG-mapper or download its
large reference databases. Keep the first FASTA header token identical to the
eggNOG `query` ID. All joins are exact; the tool reports matched row counts.
Both query annotations and neighbour annotations remain separate columns.
`row_present=0` means no row was supplied for that ID, not proof of no homology.

For GPU use, install a CUDA build of PyTorch instead of the CPU-only packages
above, then add `--device cuda` to `embed`. CPU is the default. Use
`--exclude-self` for searches against the same collection (excludes matching IDs,
not duplicate sequences under other IDs). Existing output files are never replaced.

For just searching previously generated embeddings: `pip install .` needs only
NumPy. Artifacts from other tools must be regenerated: model size alone cannot
establish compatible weights, pooling or software versions.

Try the synthetic example:

```bash
esmc-similarity embed examples/proteins.fasta toy.npz
esmc-similarity search toy.npz toy.npz hits.tsv --top-k 2 --exclude-self \
  --query-annotations examples/toy.emapper.annotations
```

The example annotations are invented parser fixtures, not biological findings.

## What the numbers mean

eggNOG-mapper supplies orthology-based annotation evidence. ESM-C represents each
protein as a vector; cosine measures the angle between two vectors. A score near
1 means similar vector directions. It is **not** a probability, sequence identity,
E-value, or confirmation of shared function. We report candidate neighbours and
their evidence without automatically transferring their labels.

In graph terms, proteins are nodes and each reported neighbour is a directed
edge weighted by cosine. A top-k graph depends on the reference collection and k;
its edges do not establish molecular interactions or causation.

See [DECISIONS.md](DECISIONS.md) for the reasoning and validation boundary.

## Reproducibility and limits

- Only the local 300M checkpoint is supported; its revision and SHA-256 are fixed
  and the downloaded weights are checked before loading.
- Final-layer residue embeddings are mean-pooled in float32, excluding BOS/EOS.
  Proteins are processed individually without padding. Long sequences are rejected
  above the default 2,048-residue resource guard; `--max-length` can change it.
  Nothing is silently truncated. Raising the guard increases memory/runtime and
  does not establish model accuracy at that length.
- Blank sequences, duplicate IDs, gaps, stop symbols and masks are rejected.
  Lowercase residues are uppercased; X/B/Z/U/O are accepted. J is unsupported.
- Saved NPZ files contain IDs, sequence hashes, vectors and model/pooling/software
  metadata. Loading disables pickle. Searches reject incompatible provenance.
- Search is exact and holds the reference embeddings in memory. Similarities are
  computed one query at a time; tied scores follow reference FASTA order. This is
  a small-collection tool, not a billion-protein index.
- CPU and CUDA floating point differences can change near-tied rankings.

## License and sources

This wrapper is MIT licensed. **Model and dependency licenses are separate.**
Weights are downloaded from their provider, not redistributed in this repository.
The pinned checkpoint uses the legacy `esm==3.1.3` interface, deliberately fixed
for reproducibility with existing embeddings; it is not the latest SDK.
Review the [checkpoint model card](https://huggingface.co/biohub/esmc-300m-2024-12)
and [ESM licensing information](https://github.com/Biohub/esm) before model use or
redistributing model-derived outputs. The provider also publishes the
[Cambrian Open License](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)
for ESM-C 300M. Free local inference does not mean every associated license has
identical terms. No blanket commercial-use claim is made here.

The wrapper is an independent implementation of the public
[ESM-C SDK interface](https://github.com/Biohub/esm/blob/v3.1.3/esm/models/esmc.py)
and [eggNOG-mapper output format](https://github.com/eggnogdb/eggnog-mapper/blob/master/eggnogmapper/annotation/output.py).
No project datasets, lab deployment scripts or model weights are included.

## Develop

```bash
pip install '.[test]'
python -m pytest -q
```

The unit suite needs neither model weights nor a GPU. See [EVIDENCE.md](EVIDENCE.md)
for the separate real-model smoke test and exactly what it demonstrates.
