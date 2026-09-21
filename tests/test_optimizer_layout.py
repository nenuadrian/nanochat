"""CPU tests for depth-wise transformer optimizer assignment."""

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.grow import grow_depth
from nanochat.optim import MuonAdamW, resolve_layer_optimizers


def make_model(depth=6):
    model = GPT(GPTConfig(
        sequence_len=64, vocab_size=128, n_layer=depth,
        n_head=2, n_kv_head=2, n_embd=64,
    ))
    model.init_weights()
    return model


def test_sophia_middle_keeps_outer_blocks_on_default_optimizer():
    layout = resolve_layer_optimizers("sophia-middle", n_layer=20, default="muon")
    assert layout[:6] == ("muon",) * 6
    assert layout[6:14] == ("sophia",) * 8
    assert layout[14:] == ("muon",) * 6


def test_explicit_layout_supports_all_three_matrix_optimizers():
    model = make_model()
    optimizer = model.setup_optimizer(
        layer_optimizers="muon*2,sophia*2,adamw*2",
        matrix_lr=0.02,
        sophia_lr=1e-4,
        sophia_batch_size=8,
    )

    matrix_groups = [group for group in optimizer.param_groups if group.get("is_matrix")]
    covered = {}
    for group in matrix_groups:
        for layer_idx in group["layer_indices"]:
            covered.setdefault(layer_idx, set()).add(group["kind"])
    assert covered == {
        0: {"muon"}, 1: {"muon"},
        2: {"sophia"}, 3: {"sophia"},
        4: {"adamw"}, 5: {"adamw"},
    }
    sophia_groups = [group for group in matrix_groups if group["kind"] == "sophia"]
    assert sophia_groups
    assert all(group["batch_size"] == 8 for group in sophia_groups)
    assert all(group["hessian_update_interval"] == 10 for group in sophia_groups)


@pytest.mark.parametrize("spec", ["sophia*5", "muon*2,nope*4", "muon*0,sophia*6"])
def test_invalid_layout_is_rejected(spec):
    with pytest.raises(ValueError):
        resolve_layer_optimizers(spec, n_layer=6)


@pytest.mark.parametrize("kind", ["adamw", "sophia"])
def test_grown_blocks_inherit_the_adjacent_matrix_optimizer(kind):
    model = make_model()
    optimizer = model.setup_optimizer(
        matrix_optimizer=kind,
        sophia_batch_size=8,
    )
    grow_depth(model, optimizer, 2, position="middle")
    grown_groups = [group for group in optimizer.param_groups if group.get("grow_tag") == "new_matrix"]
    assert grown_groups
    assert {group["kind"] for group in grown_groups} == {kind}


def test_sampled_label_hessian_update_and_metrics_work_on_cpu():
    """The extra Sophia curvature path must not reuse the normal task gradient."""
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = MuonAdamW([dict(
        kind="sophia", params=[p], lr=1e-3, betas=(0.965, 0.9), rho=0.04,
        batch_size=16, hessian_update_interval=10, eps=1e-15, weight_decay=0.0,
        layer_indices=(2, 3),
    )])
    p.grad = torch.tensor([1.0, -2.0])
    optimizer.update_sophia_hessian()

    torch.testing.assert_close(optimizer.state[p]["hessian"], torch.tensor([0.1, 0.4]))
    metrics = optimizer.sophia_metrics()
    assert metrics["sophia/layer_count"] == 2
    assert metrics["sophia/hessian_updates"] == 1
    assert metrics["sophia/effective_lr"] == pytest.approx(1e-3)
    assert metrics["sophia/curvature_mean"] == pytest.approx(0.25)
    assert metrics["sophia/curvature_rms"] == pytest.approx((0.085) ** 0.5)
    assert metrics["sophia/update_rms"] == 0.0
    assert metrics["sophia/clipped_fraction"] == 0.0
