import csv
import json

import numpy as np
import pytest

import esmc_similarity as tool


def archive(tmp_path, name, ids, vectors, **metadata):
    path = tmp_path / name
    meta = dict(format_version=1, model="test", weights_sha256="test-checkpoint",
                pooling="test-pooling", esm_version="test-version")
    meta.update(metadata)
    np.savez(path, ids=np.array(ids), vectors=np.array(vectors), metadata=json.dumps(meta))
    return path


def test_pool_excludes_special_tokens():
    tokens = np.array([[[100, 100], [1, 3], [3, 5], [-100, 100]]])
    np.testing.assert_array_equal(tool.pool_residues(tokens, 2), [2, 4])
    with pytest.raises(ValueError, match="token layout"):
        tool.pool_residues(tokens, 3)


@pytest.mark.parametrize("text", ["", "ACD", ">a\nACD\n>a\nDEF", ">\nACD",
                                    ">a\n", ">a\nAC*", ">a\nAC-D", ">a\n<mask>"])
def test_bad_fasta(tmp_path, text):
    path = tmp_path / "input.fa"
    path.write_text(text)
    with pytest.raises(ValueError):
        tool.read_fasta(path)


def test_fasta_no_truncation(tmp_path):
    path = tmp_path / "input.fa"
    path.write_text(">a description\nac\ndx\n")
    assert tool.read_fasta(path) == [("a", "ACDX")]
    with pytest.raises(ValueError, match="no truncation"):
        tool.read_fasta(path, max_length=3)


@pytest.mark.parametrize("vectors", [[[0, 0]], [[1, np.nan]], [[np.inf, 1]], [], [1, 2],
                                    [[1+2j, 3]], [["1", "2"]]])
def test_bad_vectors(vectors):
    with pytest.raises(ValueError):
        tool.normalize(vectors)


def test_cosine_scale_invariance_and_extremes():
    x = tool.normalize([[1e308, 1e308], [1e-300, 1e-300], [-1, -1], [1, -1]])
    np.testing.assert_allclose(x @ x[0], [1, 1, -1, 0], atol=1e-15)


def test_search_ranking_annotations_and_self(tmp_path):
    q = archive(tmp_path, "q.npz", ["a", "c"], [[1, 0], [0, 1]])
    r = archive(tmp_path, "r.npz", ["a", "b", "c"], [[2, 0], [1, 1], [0, 2]])
    ann = tmp_path / "annotations.tsv"
    ann.write_text("## eggNOG fixture\n#query\tseed_ortholog\tDescription\n"
                   "a\tseed1\tExample\n## footer\n")
    out = tmp_path / "hits.tsv"
    tool.search(q, r, out, top_k=9, query_annotations=ann,
                reference_annotations=ann, exclude_self=True)
    with out.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(x["query_id"], x["reference_id"]) for x in rows] == [
        ("a", "b"), ("a", "c"), ("c", "b"), ("c", "a")]
    assert float(rows[0]["cosine"]) == pytest.approx(2**-.5)
    assert rows[0]["query_eggnog_Description"] == "Example"
    assert rows[0]["reference_eggnog_row_present"] == "0"
    assert rows[3]["reference_eggnog_seed_ortholog"] == "seed1"
    with pytest.raises(FileExistsError):
        tool.search(q, r, out)


@pytest.mark.parametrize("key", ["model", "weights_sha256", "pooling", "esm_version"])
def test_provenance_mismatch(tmp_path, key):
    q = archive(tmp_path, "q.npz", ["a"], [[1, 0]])
    r = archive(tmp_path, "r.npz", ["b"], [[1, 0]], **{key: "different"})
    with pytest.raises(ValueError, match="Incompatible"):
        tool.search(q, r, tmp_path / "out.tsv")


@pytest.mark.parametrize("text", ["a\tb\n", "#query\tseed_ortholog\na\n",
    "#query\tseed_ortholog\na\ts\na\ts\n", "#query\tseed_ortholog\n\ts\n"])
def test_bad_eggnog(tmp_path, text):
    path = tmp_path / "ann.tsv"
    path.write_text(text)
    with pytest.raises(ValueError):
        tool.read_eggnog(path)


def test_id_mismatch_is_not_silently_ignored(tmp_path):
    q = archive(tmp_path, "q.npz", ["a"], [[1, 0]])
    ann = tmp_path / "ann.tsv"
    ann.write_text("#query\tseed_ortholog\nwrong\ts\n")
    with pytest.raises(ValueError, match="No query eggNOG IDs match"):
        tool.search(q, q, tmp_path / "out.tsv", query_annotations=ann)


def test_archive_validation(tmp_path):
    path = archive(tmp_path, "bad.npz", ["a", "a"], [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="unique"):
        tool.load_embeddings(path)
    path = archive(tmp_path, "empty.npz", ["a"], [[1, 0]], pooling="")
    with pytest.raises(ValueError, match="provenance"):
        tool.load_embeddings(path)
    np.savez(tmp_path / "pickle.npz", ids=np.array([{}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        tool.load_embeddings(tmp_path / "pickle.npz")


def test_ties_keep_reference_order(tmp_path):
    q = archive(tmp_path, "q.npz", ["q"], [[1, 0]])
    r = archive(tmp_path, "r.npz", ["z", "a"], [[1, 0], [2, 0]])
    out = tmp_path / "out.tsv"
    tool.search(q, r, out, top_k=1)
    with out.open() as handle:
        assert next(csv.DictReader(handle, delimiter="\t"))["reference_id"] == "z"
