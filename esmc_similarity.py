"""Local protein embeddings and cosine neighbours. Built with ESM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

MODEL = "esmc_300m"
MODEL_REPO = "EvolutionaryScale/esmc-300m-2024-12"
REVISION = "7f10b20ae75017b2dbc884070e03434515709a8d"
WEIGHTS = "data/weights/esmc_300m_2024_12_v0.pth"
WEIGHTS_SHA256 = "323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea"
POOLING = "final-layer-residue-mean-float32-v1"
ANNOTATIONS = ("seed_ortholog", "evalue", "score", "COG_category", "Description",
               "Preferred_name", "GOs", "EC", "KEGG_ko", "PFAMs")


def read_fasta(path, max_length=2048):
    """Keep the first header token as ID; reject malformed or lossy inputs."""
    records, seen = [], set()
    name, parts = None, []

    def finish():
        if name is None:
            return
        sequence = "".join(parts).upper()
        if not sequence or len(sequence) > max_length:
            raise ValueError(f"{name}: sequence must contain 1–{max_length} residues; no truncation")
        if set(sequence) - set("ACDEFGHIKLMNPQRSTVWYXBZUO"):
            raise ValueError(f"{name}: invalid residues (gaps, stops and masks are unsupported)")
        records.append((name, sequence))

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                words = line[1:].split()
                if not words or words[0] in seen:
                    raise ValueError("FASTA IDs must be nonempty and unique")
                name, parts = words[0], []
                seen.add(name)
            elif name is None:
                raise ValueError("FASTA sequence precedes its header")
            else:
                parts.append(line)
    finish()
    if not records:
        raise ValueError("FASTA is empty")
    return records


def normalize(vectors):
    raw = np.asarray(vectors)
    if raw.dtype.kind not in "fiu":
        raise ValueError("Embeddings must contain real numbers")
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or 0 in vectors.shape or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a nonempty finite matrix")
    scale = np.max(np.abs(vectors), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Zero embeddings have undefined cosine similarity")
    scaled = vectors / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def pool_residues(embeddings, length):
    """One unpadded sequence: BOS, L residues, EOS. Never average BOS/EOS."""
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[1] != length + 2:
        raise ValueError("Unexpected ESM-C token layout")
    if length < 1 or not np.isfinite(array).all():
        raise ValueError("Invalid residue embeddings")
    return array[0, 1:-1].mean(axis=0, dtype=np.float32)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embed(fasta, output, device="cpu", max_length=2048, threads=2):
    records = read_fasta(fasta, max_length)
    if threads < 1:
        raise ValueError("threads must be positive")
    if importlib.metadata.version("esm") != "3.1.3":
        raise ValueError("Install the tested backend: pip install 'esmc-similarity[embed]'")
    import torch
    from huggingface_hub import hf_hub_download
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers
    from esm.sdk.api import ESMProtein, LogitsConfig

    torch.set_num_threads(threads)
    # Only a weight download. User sequences never go to an inference service.
    checkpoint = hf_hub_download(MODEL_REPO, WEIGHTS, revision=REVISION)
    if sha256_file(checkpoint) != WEIGHTS_SHA256:
        raise ValueError("Checkpoint checksum mismatch")
    model = ESMC(d_model=960, n_heads=15, n_layers=30,
                 tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    vectors = []
    with torch.inference_mode():
        for index, (name, sequence) in enumerate(records, 1):
            encoded = model.encode(ESMProtein(sequence=sequence))
            result = model.logits(encoded, LogitsConfig(return_embeddings=True))
            vectors.append(pool_residues(result.embeddings.float().cpu().numpy(), len(sequence)))
            print(f"Embedded {index}/{len(records)}", file=sys.stderr)
    matrix = np.stack(vectors)
    normalize(matrix)  # Validate before writing anything.
    metadata = dict(format_version=1, model=MODEL, model_repo=MODEL_REPO,
                    revision=REVISION, weights_sha256=WEIGHTS_SHA256, pooling=POOLING,
                    esm_version=importlib.metadata.version("esm"),
                    torch_version=torch.__version__, device=device, max_length=max_length)
    with open(output, "xb") as handle:
        np.savez_compressed(handle, ids=np.array([r[0] for r in records]), vectors=matrix,
                            sequence_sha256=np.array([hashlib.sha256(r[1].encode()).hexdigest()
                                                      for r in records]),
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))


def load_embeddings(path):
    with np.load(path, allow_pickle=False) as archive:
        ids, vectors = archive["ids"], archive["vectors"]
        meta = json.loads(str(archive["metadata"].item()))
    if (ids.ndim != 1 or ids.dtype.kind != "U" or len(set(ids)) != len(ids)
            or any(not x or any(c.isspace() for c in x) for x in ids)):
        raise ValueError("Embedding IDs must be unique nonempty strings without whitespace")
    vectors = normalize(vectors)
    if len(ids) != len(vectors):
        raise ValueError("Embedding ID/vector count mismatch")
    required = ("model", "weights_sha256", "pooling", "esm_version")
    if (not isinstance(meta, dict) or meta.get("format_version") != 1
            or any(not isinstance(meta.get(k), str) or not meta[k] for k in required)):
        raise ValueError("Missing or unsupported embedding provenance")
    return ids, vectors, meta


def read_eggnog(path):
    """Read eggNOG-mapper's #query header and preserve selected evidence fields."""
    if path is None:
        return {}
    header, records = None, {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith("#query\t") or line.startswith("query\t"):
                if header is not None:
                    raise ValueError("Repeated eggNOG header")
                header = line.lstrip("#").split("\t")
                if len(set(header)) != len(header) or "seed_ortholog" not in header:
                    raise ValueError("Malformed eggNOG header")
            elif not line or line.startswith("#"):
                continue
            else:
                if header is None:
                    raise ValueError("eggNOG file needs its #query header")
                values = line.split("\t")
                if len(values) != len(header):
                    raise ValueError("eggNOG row/header width mismatch")
                row = dict(zip(header, values))
                name = row["query"]
                if not name or name in records:
                    raise ValueError("eggNOG query IDs must be unique and nonempty")
                records[name] = row
    if header is None:
        raise ValueError("eggNOG file needs its #query header")
    return records


def search(queries, reference, output, top_k=5, query_annotations=None,
           reference_annotations=None, exclude_self=False):
    if top_k < 1:
        raise ValueError("top-k must be positive")
    qids, q, qm = load_embeddings(queries)
    rids, r, rm = load_embeddings(reference)
    for key in ("model", "weights_sha256", "pooling", "esm_version"):
        if qm[key] != rm[key]:
            raise ValueError(f"Incompatible embeddings: {key}")
    if q.shape[1] != r.shape[1]:
        raise ValueError("Embedding dimensions differ")
    qa, ra = read_eggnog(query_annotations), read_eggnog(reference_annotations)
    for label, ann, ids in (("query", qa, qids), ("reference", ra, rids)):
        matched = len(set(ids) & ann.keys())
        print(f"{label} eggNOG rows matched: {matched}/{len(ids)}", file=sys.stderr)
        if ann and not matched:
            raise ValueError(f"No {label} eggNOG IDs match; IDs are joined exactly")
    fields = ["query_id", "reference_id", "rank", "cosine"]
    for prefix in ("query_eggnog_", "reference_eggnog_"):
        fields += [prefix + "row_present"] + [prefix + x for x in ANNOTATIONS]
    with open(output, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for name, vector in zip(qids, q):
            # One query at a time: no Q x R similarity matrix retained.
            scores = np.clip(r @ vector, -1, 1)
            eligible = np.flatnonzero(rids != name) if exclude_self else np.arange(len(rids))
            # Stable ties follow reference FASTA order.
            best = eligible[np.argsort(-scores[eligible], kind="stable")[:top_k]]
            for rank, index in enumerate(best, 1):
                hit = rids[index]
                row = dict(query_id=name, reference_id=hit, rank=rank,
                           cosine=format(scores[index], ".9g"))
                for prefix, ann, identifier in (("query_eggnog_", qa, name),
                                                 ("reference_eggnog_", ra, hit)):
                    row[prefix + "row_present"] = int(identifier in ann)
                    row.update({prefix + k: ann.get(identifier, {}).get(k, "")
                                for k in ANNOTATIONS})
                writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    embeddings = commands.add_parser("embed", help="FASTA → local ESM-C 300M embeddings")
    embeddings.add_argument("fasta")
    embeddings.add_argument("output")
    embeddings.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    embeddings.add_argument("--max-length", type=int, default=2048)
    embeddings.add_argument("--threads", type=int, default=2)
    neighbors = commands.add_parser("search", help="Rank cosine neighbours; optionally join eggNOG")
    neighbors.add_argument("queries")
    neighbors.add_argument("reference")
    neighbors.add_argument("output")
    neighbors.add_argument("--top-k", type=int, default=5)
    neighbors.add_argument("--query-annotations")
    neighbors.add_argument("--reference-annotations")
    neighbors.add_argument("--exclude-self", action="store_true", help="Exclude identical IDs")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    # Fail early, before loading weights, if output already exists.
    if Path(args["output"]).exists():
        parser.error("Output exists; choose a new filename")
    try:
        (embed if command == "embed" else search)(**args)
    except (ValueError, OSError, KeyError, ImportError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
