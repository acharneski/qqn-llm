"""Verify PyTree-native QQN matches flat-array QQN numerically and works in bf16."""

import jax
import jax.numpy as jnp
from qqn_jax import QQN, AdamOracle, Fallback, LBFGSOracle, SecantOracle


def test_pytree_qqn_matches_flat_lbfgs():
    """PyTree QQN with LBFGSOracle must match flat-vector QQN on a quadratic bowl."""

    def flat_fun(x):
        a = x[:3]
        b = x[3:].reshape(2, 2)
        return jnp.sum((a - 1.0) ** 2) + 2.0 * jnp.sum((b + 0.5) ** 2)

    def tree_fun(p):
        return jnp.sum((p["a"] - 1.0) ** 2) + 2.0 * jnp.sum((p["b"] + 0.5) ** 2)

    x0 = jnp.array([0.2, -0.3, 0.7, 0.5, -0.1, 0.4, -0.8], dtype=jnp.float32)
    p0 = {"a": x0[:3], "b": x0[3:].reshape(2, 2)}

    s_flat = QQN(flat_fun, maxiter=10, history_size=5, oracle="lbfgs", line_search="backtracking")
    s_tree = QQN(tree_fun, maxiter=10, history_size=5, oracle="lbfgs", line_search="backtracking")

    xf, st_f = s_flat.run(x0)
    pt, st_t = s_tree.run(p0)

    assert jnp.allclose(st_f.value, st_t.value, atol=1e-5)
    assert jnp.allclose(xf[:3], pt["a"], atol=1e-4)
    assert jnp.allclose(xf[3:].reshape(2, 2), pt["b"], atol=1e-4)
    assert float(st_t.value) < 1e-4


def test_pytree_qqn_bfloat16_preserves_dtype_and_descends():
    """Mixed-precision bf16 weights must stay bf16 and decrease loss cleanly."""

    def tree_fun(p):
        w = p["w"].astype(jnp.float32)
        b = p["b"].astype(jnp.float32)
        return jnp.mean((w - 0.5) ** 2) + jnp.mean((b + 0.25) ** 2)

    p0 = {
        "w": jnp.zeros((4, 4), dtype=jnp.bfloat16),
        "b": jnp.ones((4,), dtype=jnp.bfloat16),
    }

    for oracle in (
        LBFGSOracle(history_size=4),
        AdamOracle(learning_rate=0.1),
        SecantOracle(),
        Fallback([LBFGSOracle(history_size=4), SecantOracle()]),
    ):
        solver = QQN(tree_fun, maxiter=8, oracle=oracle, line_search="backtracking")
        pf, st = solver.run(p0)
        assert pf["w"].dtype == jnp.bfloat16
        assert pf["b"].dtype == jnp.bfloat16
        assert jnp.isfinite(st.value)
        assert float(st.value) < float(tree_fun(p0))


def test_refresh_state_consistent_batch():
    """QQN.refresh_state updates value/grad for a new batch while keeping oracle history."""

    def batch_fun(p, target):
        return jnp.mean((p["w"] - target) ** 2)

    p0 = {"w": jnp.zeros((4,), dtype=jnp.float32)}
    b1 = jnp.ones((4,), dtype=jnp.float32)
    b2 = jnp.full((4,), 2.0, dtype=jnp.float32)

    solver = QQN(batch_fun, maxiter=1, oracle="lbfgs", history_size=4, line_search="backtracking")
    st = solver.init_state(p0, b1)
    p1, st = solver.update(p0, st, b1)
    st_ref = solver.refresh_state(p1, st, b2)

    expected_val = float(batch_fun(p1, b2))
    assert jnp.isclose(st_ref.value, expected_val, atol=1e-5)
    assert int(st_ref.num_evals) == int(st.num_evals) + 1
    p2, st2 = solver.update(p1, st_ref, b2)
    assert float(st2.value) < expected_val


def test_lbfgs_adam_handles_anisotropic_scales():
    """Adam-preconditioned L-BFGS must converge rapidly on ill-conditioned multi-scale PyTrees."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import numpy as np
    from qqn_llm.config import OptimizerConfig
    from qqn_llm.optimizers.qqn_adapter import QQNRunner

    # 10,000x curvature ratio between leaves (mimics transformer embedder vs MLP scale anisotropy)
    def aniso_loss(p, _batch):
        return 1e-2 * jnp.sum((p["slow"] - 1.0) ** 2) + 1e2 * jnp.sum((p["fast"] + 0.5) ** 2)

    mesh = Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    sharding = NamedSharding(mesh, P())
    p0 = {
        "slow": jnp.zeros((8,), dtype=jnp.float32),
        "fast": jnp.ones((8,), dtype=jnp.float32),
    }

    cfg = OptimizerConfig(
        name="qqn_lbfgs_adam_m5",
        kind="qqn",
        oracle="lbfgs_adam",
        learning_rate=0.2,
        history_size=5,
        ls_max_iter=5,
        max_t=2.0,
    )
    runner = QQNRunner(aniso_loss, cfg, mesh=mesh, param_sharding=sharding)
    st = runner.init(p0, None)
    init_loss = st.value
    p = p0
    for _ in range(15):
        p, st, _ = runner.step(p, st, None)

    assert st.value < init_loss * 1e-3, f"Expected >1000x loss drop, got {init_loss} -> {st.value}"


def test_linesearch_and_bounded_spline_variants():
    """Verify Strong Wolfe, Bounded Hermite Spline, Linear chord, and Null search all descend."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import numpy as np
    from qqn_llm.config import OptimizerConfig
    from qqn_llm.optimizers.qqn_adapter import QQNRunner

    def rosen_bowl(p, _batch):
        x = p["x"]
        return jnp.sum(10.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    mesh = Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    sharding = NamedSharding(mesh, P())
    p0 = {"x": jnp.array([-0.5, 0.5, -0.5, 0.5], dtype=jnp.float32)}

    variants = [
        ("armijo", "backtracking", "quadratic", 1.0),
        ("strong_wolfe", "strong_wolfe", "quadratic", 2.0),
        ("bounded_spline", "backtracking", "spline", 2.0),
        ("linear_chord", "backtracking", "linear", 1.0),
        ("null_search", "null", "quadratic", 1.0),
    ]
    for name, ls, path, max_t in variants:
        cfg = OptimizerConfig(
            name=name,
            kind="qqn",
            oracle="lbfgs_adam",
            learning_rate=0.05,
            history_size=5,
            line_search=ls,
            path_strategy=path,
            c1=1e-4 if ls == "strong_wolfe" else 1e-6,
            c2=0.9,
            ls_max_iter=4,
            max_t=max_t,
            spline_refine_rounds=2,
            spline_max_control_points=6,
        )
        runner = QQNRunner(rosen_bowl, cfg, mesh=mesh, param_sharding=sharding)
        st = runner.init(p0, None)
        init_loss = st.value
        p = p0
        for _ in range(10):
            p, st, _ = runner.step(p, st, None)
        assert jnp.isfinite(st.value), f"{name} produced non-finite loss"
        assert st.value < init_loss * 0.5, f"{name} failed to descend: {init_loss} -> {st.value}"
        # Bounded spline must stay within tight eval budget (<= (ls_max_iter + rounds + 1) per step)
        if path == "spline":
            assert st.num_evals <= 1 + 10 * (cfg.ls_max_iter + cfg.spline_refine_rounds + 1)


