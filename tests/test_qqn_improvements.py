"""Targeted tests verifying QQN mathematical, algorithmic, and systems improvements."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from qqn_llm.config import OptimizerConfig
from qqn_llm.optimizers.baselines import AdamWRunner
from qqn_llm.optimizers.qqn_adapter import QQNRunner
from qqn_llm.train import (
    _compute_weight_deviation,
    _make_device_weight_deviation_fn,
)


def _single_device_mesh_and_sharding():
    mesh = Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    return mesh, NamedSharding(mesh, P())


def test_tangent_modes_separate_reparameterization_from_metric_change():
    """Verify 'raw', path-invariant 'reparam', 'scaled', and 'preconditioned' modes."""
    mesh, sharding = _single_device_mesh_and_sharding()

    def llm_scale_bowl(p, _batch):
        return 5e4 * jnp.sum((p["embed"] - 4e-5) ** 2) + 5.0 * jnp.sum(
            (p["mlp"] + 4e-5) ** 2
        )

    p0 = {
        "embed": jnp.zeros((16,), dtype=jnp.float32),
        "mlp": jnp.zeros((16,), dtype=jnp.float32),
    }

    # 1. Canonical 'raw' QQN contains small descent steps near t ~ tau, while t=0.5 overshoots
    cfg_raw = OptimizerConfig(
        name="raw_path",
        kind="qqn",
        oracle="lbfgs_adam",
        learning_rate=2e-5,
        tangent_mode="raw",
        grad_clip_norm=0.0,
    )
    runner_raw = QQNRunner(llm_scale_bowl, cfg_raw, mesh=mesh, param_sharding=sharding)
    st_raw = runner_raw.init(p0, None)
    qn_raw, g_pre_raw, _ = runner_raw._sharded_lbfgs_direction_and_tangent(
        st_raw.grad, st_raw.oracle_state
    )
    g_tan_raw, tau_raw, _, _ = runner_raw._prepare_step_jit(
        st_raw.grad, g_pre_raw, qn_raw
    )
    cand_raw_half = runner_raw._candidate_jit(
        p0, g_tan_raw, qn_raw, jnp.asarray(0.5, jnp.float32), tau_raw
    )
    assert float(llm_scale_bowl(cand_raw_half, None)) > st_raw.value * 1e4

    # 2. 'reparam' traverses the EXACT canonical 'raw' path locus via Möbius t(s),
    # while un-biasing s=0.5 so dyadic probes stay in the descending step-norm regime!
    cfg_rep = OptimizerConfig(
        name="reparam_path",
        kind="qqn",
        oracle="lbfgs_adam",
        learning_rate=2e-5,
        tangent_mode="reparam",
        grad_clip_norm=0.0,
    )
    runner_rep = QQNRunner(llm_scale_bowl, cfg_rep, mesh=mesh, param_sharding=sharding)
    st_rep = runner_rep.init(p0, None)
    qn_rep, g_pre_rep, _ = runner_rep._sharded_lbfgs_direction_and_tangent(
        st_rep.grad, st_rep.oracle_state
    )
    g_tan_rep, tau_rep, slope0_rep, _ = runner_rep._prepare_step_jit(
        st_rep.grad, g_pre_rep, qn_rep
    )
    assert float(slope0_rep) < 0.0
    for s_val in (0.1, 0.5, 0.9, 1.0):
        tau_f = float(tau_rep)
        t_equiv = (tau_f * s_val) / ((1.0 - s_val) + tau_f * s_val)
        p_rep = runner_rep._candidate_jit(
            p0, g_tan_rep, qn_rep, jnp.asarray(s_val, jnp.float32), tau_rep
        )
        p_raw_at_t = runner_raw._candidate_jit(
            p0, g_tan_raw, qn_raw, jnp.asarray(t_equiv, jnp.float32), tau_raw
        )
        for k in ("embed", "mlp"):
            assert jnp.allclose(p_rep[k], p_raw_at_t[k], rtol=1e-5, atol=1e-7)
    cand_rep_half = runner_rep._candidate_jit(
        p0, g_tan_rep, qn_rep, jnp.asarray(0.5, jnp.float32), tau_rep
    )
    assert float(llm_scale_bowl(cand_rep_half, None)) < st_rep.value

    # 3. 'scaled' and 'preconditioned' both descend smoothly across s in (0, 1]
    for mode in ("scaled", "preconditioned"):
        cfg_m = OptimizerConfig(
            name=f"{mode}_path",
            kind="qqn",
            oracle="lbfgs_adam",
            learning_rate=2e-5,
            tangent_mode=mode,
            grad_clip_norm=1.0,
        )
        runner_m = QQNRunner(llm_scale_bowl, cfg_m, mesh=mesh, param_sharding=sharding)
        st_m = runner_m.init(p0, None)
        qn_m, g_pre_m, _ = runner_m._sharded_lbfgs_direction_and_tangent(
            st_m.grad, st_m.oracle_state
        )
        g_tan_m, tau_m, slope0_m, _ = runner_m._prepare_step_jit(
            st_m.grad, g_pre_m, qn_m
        )
        assert float(slope0_m) < 0.0
        cand_half = runner_m._candidate_jit(
            p0, g_tan_m, qn_m, jnp.asarray(0.5, jnp.float32), tau_m
        )
        assert float(llm_scale_bowl(cand_half, None)) < st_m.value


def test_outer_lr_schedule_scales_lbfgs_secant_subspace():
    """Cosine lr_mult must scale the full L-BFGS direction (including rho*s*s^T), not only H0."""
    mesh, sharding = _single_device_mesh_and_sharding()

    def quad(p, _b):
        return 0.5 * jnp.sum(jnp.arange(1.0, 9.0) * (p["w"] - 1.0) ** 2)

    p0 = {"w": jnp.zeros((8,), dtype=jnp.float32)}
    cfg = OptimizerConfig(
        name="sched_lbfgs",
        kind="qqn",
        oracle="lbfgs_adam",
        learning_rate=0.05,
        history_size=4,
        use_lr_schedule=True,
        max_qn_gain=0.0,  # disable gain cap to test pure lr_mult linearity
    )
    runner = QQNRunner(quad, cfg, mesh=mesh, param_sharding=sharding, total_steps=10)
    st = runner.init(p0, None)
    p = p0
    for _ in range(3):
        p, st, _ = runner.step(p, st, None)
    assert len(st.oracle_state.pairs) >= 2

    d_full, _, _ = runner._sharded_lbfgs_direction_and_tangent(
        st.grad, st.oracle_state, lr_mult=1.0
    )
    d_tenth, _, _ = runner._sharded_lbfgs_direction_and_tangent(
        st.grad, st.oracle_state, lr_mult=0.1
    )
    assert jnp.allclose(d_tenth["w"], 0.1 * d_full["w"], rtol=1e-5, atol=1e-6)


def test_spike_safe_v_conditioning_resists_step1_gradient_shock():
    """A 100x step-1 gradient spike must not freeze subsequent steps when grad_clip_norm > 0."""
    mesh, sharding = _single_device_mesh_and_sharding()

    def batch_loss(p, scale):
        return scale * 0.5 * jnp.sum((p["w"] - 0.01) ** 2)

    p0 = {"w": jnp.zeros((32,), dtype=jnp.float32)}

    # Unclipped beta2=0.999 freezes after a 100x step-1 spike
    cfg_unclipped = OptimizerConfig(
        name="unclipped",
        kind="qqn",
        oracle="lbfgs_adam",
        learning_rate=1e-3,
        beta2=0.999,
        grad_clip_norm=0.0,
        clear_lbfgs_on_refresh=True,
    )
    r_un = QQNRunner(batch_loss, cfg_unclipped, mesh=mesh, param_sharding=sharding)
    st_un = r_un.init(p0, jnp.asarray(1e4, jnp.float32))
    p_un, st_un, _ = r_un.step(p0, st_un, jnp.asarray(1e4, jnp.float32))
    st_un = r_un.refresh(p_un, st_un, jnp.asarray(1.0, jnp.float32))
    p_un2, _, _ = r_un.step(p_un, st_un, jnp.asarray(1.0, jnp.float32))
    step2_move_unclipped = float(jnp.linalg.norm(p_un2["w"] - p_un["w"]))

    # Clipped H0_clipped maintains clean ~lr step scale on step 2
    cfg_clipped = OptimizerConfig(
        name="clipped",
        kind="qqn",
        oracle="lbfgs_adam",
        learning_rate=1e-3,
        beta2=0.98,
        grad_clip_norm=1.0,
        clear_lbfgs_on_refresh=True,
    )
    r_cl = QQNRunner(batch_loss, cfg_clipped, mesh=mesh, param_sharding=sharding)
    st_cl = r_cl.init(p0, jnp.asarray(1e4, jnp.float32))
    p_cl, st_cl, _ = r_cl.step(p0, st_cl, jnp.asarray(1e4, jnp.float32))
    st_cl = r_cl.refresh(p_cl, st_cl, jnp.asarray(1.0, jnp.float32))
    p_cl2, _, _ = r_cl.step(p_cl, st_cl, jnp.asarray(1.0, jnp.float32))
    step2_move_clipped = float(jnp.linalg.norm(p_cl2["w"] - p_cl["w"]))

    assert step2_move_clipped > 20.0 * step2_move_unclipped


def test_strong_wolfe_tracks_cur_v_separately_from_best_v():
    """Strong Wolfe zoom must bracket properly when t=1 has steep slope and t=2 overshoots."""
    mesh, sharding = _single_device_mesh_and_sharding()

    # Minimum at w = 1.35: at t=1 (w=1.0) |slope|/|slope0| = 0.35/1.35 = 0.259 > c2=0.2,
    # so Strong Wolfe extrapolates to t=2 (w=2.0, overshoot) and zooms into t=1.5 (w=1.5).
    def quad_valley(p, _b):
        w = p["w"]
        return 0.5 * jnp.sum((w - 1.35) ** 2)

    p0 = {"w": jnp.zeros((4,), dtype=jnp.float32)}
    cfg = OptimizerConfig(
        name="sw_zoom",
        kind="qqn",
        oracle="lbfgs_adam",
        learning_rate=1.0,
        grad_clip_norm=0.0,
        line_search="strong_wolfe",
        tangent_mode="reparam",
        c1=1e-4,
        c2=0.2,
        ls_max_iter=5,
        max_t=2.0,
    )
    runner = QQNRunner(quad_valley, cfg, mesh=mesh, param_sharding=sharding)
    st = runner.init(p0, None)
    p1, st1, metrics = runner.step(p0, st, None)
    assert metrics["ls_success"] == 1.0
    val_at_1 = float(quad_valley({"w": jnp.ones((4,), dtype=jnp.float32)}, None))
    assert st1.value < val_at_1 * 0.25
    assert 1.0 < metrics["step_size"] < 2.0


def test_adamw_nonzero_step1_and_two_phase_preconditioner_handoff():
    """AdamW step 1 must move weights, and exported nu must warm-start Phase-2 QQNRunner."""
    mesh, sharding = _single_device_mesh_and_sharding()

    def loss_fn(p, _b):
        return 0.5 * jnp.sum((p["w"] - 1.0) ** 2)

    p0 = {"w": jnp.zeros((16,), dtype=jnp.float32)}
    adam_cfg = OptimizerConfig(
        name="adamw",
        kind="adamw",
        learning_rate=1e-2,
        warmup_ratio=0.2,
        weight_decay=0.01,
        grad_clip_norm=1.0,
    )
    adam_runner = AdamWRunner(loss_fn, adam_cfg, total_steps=10)
    st = adam_runner.init(p0, None)
    p1, st1, _ = adam_runner.step(p0, st, None)
    # Step 1 must perform a genuine non-zero update!
    assert float(jnp.linalg.norm(p1["w"] - p0["w"])) > 0.0
    assert float(loss_fn(p1, None)) < float(loss_fn(p0, None))

    for _ in range(4):
        p1, st1, _ = adam_runner.step(p1, st1, None)

    exported = adam_runner.export_preconditioner(st1)
    assert exported is not None
    v_exp, step_exp = exported
    assert step_exp == 5
    assert float(jnp.sum(v_exp["w"])) > 0.0

    # Hand off to Phase-2 QQN lbfgs_adam
    qqn_cfg = OptimizerConfig(
        name="polish_qqn",
        kind="qqn",
        oracle="lbfgs_adam",
        learning_rate=5e-3,
    )
    qqn_runner = QQNRunner(loss_fn, qqn_cfg, mesh=mesh, param_sharding=sharding)
    qqn_st = qqn_runner.init(p1, None, warm_precond=exported)
    assert qqn_st.oracle_state.step == 5
    assert jnp.allclose(qqn_st.oracle_state.v["w"], v_exp["w"])


def test_device_weight_deviation_matches_exact_f64_and_terminal_batch_saves_evals():
    """On-device FSDP deviation matches CPU float64, and will_refresh_next skips dead vg evals."""
    mesh, sharding = _single_device_mesh_and_sharding()
    rng = np.random.default_rng(0)
    init_p = {
        "a": jax.device_put(
            jnp.asarray(rng.standard_normal((64, 32)), dtype=jnp.bfloat16), sharding
        ),
        "b": jax.device_put(
            jnp.asarray(rng.standard_normal((32,)), dtype=jnp.bfloat16), sharding
        ),
    }
    cur_p = {
        "a": init_p["a"].astype(jnp.float32) + 1e-4,
        "b": init_p["b"].astype(jnp.float32) - 2e-4,
    }
    n_params = 64 * 32 + 32
    dev_gpu = _make_device_weight_deviation_fn(init_p, mesh, n_params)(cur_p)
    dev_cpu = _compute_weight_deviation(cur_p, init_p)
    for k in ("weight_l1_dist", "weight_l2_dist", "weight_l1_mean", "weight_l2_rms"):
        assert np.isclose(dev_gpu[k], dev_cpu[k], rtol=1e-6, atol=1e-9)

    # Verify null search with will_refresh_next=True adds 0 extra vg evals
    def quad(p, _b):
        return 0.5 * jnp.sum((p["b"] - 1.0) ** 2)

    cfg_null = OptimizerConfig(
        name="null_fast",
        kind="qqn",
        oracle="lbfgs_adam",
        learning_rate=1e-2,
        line_search="null",
        clear_lbfgs_on_refresh=True,
    )
    runner = QQNRunner(quad, cfg_null, mesh=mesh, param_sharding=sharding)
    st = runner.init(cur_p, None)
    assert st.num_evals == 1
    p_next, st_next, metrics = runner.step(cur_p, st, None, will_refresh_next=True)
    assert st_next.num_evals == 1
    assert metrics["ls_success"] == 1.0
    assert float(quad(p_next, None)) < float(quad(cur_p, None))
