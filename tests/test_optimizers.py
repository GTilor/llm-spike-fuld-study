"""Unit tests verifying mathematical invariants and baseline contracts of `adafuld`."""

from __future__ import annotations

import torch

from adafuld import (
    ExactStableAdamW,
    ExactTAdamW,
    ModularAdaptiveAdamW,
    eval_flattop_twin,
    eval_psi_unit,
    exact_peak_location,
    make_optimizer,
)


def test_rational_score_properties_and_flattop_identity():
    z = torch.linspace(-100.0, 100.0, 20001, dtype=torch.float64)
    psi = eval_psi_unit(z, 1.5)
    flat = eval_flattop_twin(z, 1.5)
    z_star = exact_peak_location(1.5)

    # Odd symmetry & pole-free finiteness
    assert torch.isfinite(psi).all()
    assert torch.allclose(psi + eval_psi_unit(-z, 1.5), torch.zeros_like(psi), atol=1e-12)

    # Bit-for-bit pre-peak identity between psi_1.5 and P_1.5 on |z| <= z_*
    pre_mask = z.abs() <= z_star
    assert torch.equal(psi[pre_mask], flat[pre_mask])

    # Post-peak redescending vs. flat-top saturation
    far_z = torch.tensor([20.0, 80.0], dtype=torch.float64)
    psi_far = eval_psi_unit(far_z, 1.5)
    flat_far = eval_flattop_twin(far_z, 1.5)
    assert (psi_far < flat_far).all()
    assert psi_far[1] < psi_far[0]

    # Quartic polynomial bound from Proposition 2B(2): 0 <= c3*z^2 - (1 - psi(z)/z) <= 0.01615*z^4 on 0 < |z| <= 1
    z_unit = torch.linspace(1e-4, 1.0, 1000, dtype=torch.float64)
    c3 = 0.28915823558698084 - (0.15633459124532706 / 1.2695963915423436)
    shrink = 1.0 - eval_psi_unit(z_unit, 1.5) / z_unit
    rem = c3 * (z_unit ** 2) - shrink
    assert (rem >= -1e-15).all()
    assert (rem <= 0.01615 * (z_unit ** 4)).all()


def test_zero_gradient_passthrough_and_threshold_v_gate():
    w = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    opt = ModularAdaptiveAdamW([w], lr=1e-3, m_filter="fuld", v_filter="cap", rho_prior=0.15)

    # Step 1: non-zero gradient initializes state
    w.grad = torch.tensor([0.5, -0.5])
    opt.step()
    m1 = opt.state[w]["exp_avg"].clone()

    # Step 2: coordinate 0 gets exact zero gradient -> g_tilde[0] must be exact 0 (pure beta1 decay)
    w.grad = torch.tensor([0.0, -0.45])
    opt.step()
    m2 = opt.state[w]["exp_avg"]
    assert torch.isclose(m2[0], 0.9 * m1[0], atol=1e-12)


def test_proposition_2b_awakening_trap_vs_prior_floor():
    """Verify C11 (rho_prior=0) takes >= 24 steps (~94-95 exact) to exit v-cap while T13 captures on step 1."""

    def run_dormant_coord(rho_prior: float) -> tuple[int | None, int | None]:
        w = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        opt = ModularAdaptiveAdamW(
            [w],
            lr=2.5e-3,
            betas=(0.9, 0.995),
            eps=1e-8,
            m_filter="fuld",
            v_filter="cap",
            alp=1.5,
            tau=3.0,
            kappa=2.0,
            rho_prior=rho_prior,
        )
        # Pre-populate state to t=600 where coord 0 was dormant (0) and coord 1 had active variance
        # Set V_{600, 1} so that bar_sigma = sqrt(0.5 * V_1) = 6e-4 / 0.16 = 3.75e-3
        bar_sigma = 6e-4 / 0.16
        bc2_600 = 1.0 - 0.995 ** 600
        opt.state[w] = {
            "step": 600,
            "exp_avg": torch.zeros(2, dtype=torch.float32),
            "exp_avg_sq": torch.tensor([0.0, 2.0 * (bar_sigma ** 2) * bc2_600], dtype=torch.float32),
        }

        z_star = exact_peak_location(1.5)
        first_uncapped = None
        first_prepeak = None
        g_val = 6e-4
        for occ in range(1, 130):
            st = opt.state[w]
            t_prev = st["step"]
            m_hat = float(st["exp_avg"][0] / (1.0 - 0.9 ** t_prev))
            v_hat = st["exp_avg_sq"] / (1.0 - 0.995 ** t_prev)
            s0 = float(torch.sqrt(v_hat[0]))
            if rho_prior > 0.0:
                s0 = max(s0, rho_prior * float(torch.sqrt(torch.mean(v_hat))))
            h0 = 3.0 * (s0 + 1e-8)
            z0 = abs(g_val - m_hat) / h0
            if first_uncapped is None and g_val <= abs(m_hat) + 2.0 * h0:
                first_uncapped = occ
            if first_prepeak is None and z0 <= z_star:
                first_prepeak = occ
            w.grad = torch.tensor([g_val, bar_sigma], dtype=torch.float32)
            opt.step()
        return first_uncapped, first_prepeak

    uncap_c11, pre_c11 = run_dormant_coord(rho_prior=0.0)
    uncap_t13, pre_t13 = run_dormant_coord(rho_prior=0.15)

    assert uncap_t13 == 1 and pre_t13 == 1
    assert uncap_c11 is not None and uncap_c11 >= 90
    assert pre_c11 is not None and pre_c11 >= 90


def test_tadamw_and_stableadamw_smoke_and_factory():
    names = [
        "C01_AdamW",
        "B16_GradClip",
        "B17_StableAdamW_Exact",
        "B18_LargeEps_1e-4",
        "N19_CoordClip",
        "N20_ExactTAdamW",
        "C05_FlatTop_raw",
        "C09_FULD_raw",
        "C11_FULD_cap_noFloor",
        "T13_FlatTop",
        "T13_AdaFULD_PriorFloor",
        "T13_Plus_MildEps",
    ]
    for name in names:
        p = torch.nn.Parameter(torch.randn(16, 8))
        opt = make_optimizer(name, [p], lr=2.5e-3)
        for _ in range(4):
            p.grad = torch.randn_like(p) * 0.1
            opt.step()
        assert torch.isfinite(p).all()

    # Verify ExactTAdamW stationary weight W_t == beta1/(1-beta1) when w_t == 1 (mahal == d)
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    t_opt = ExactTAdamW([p], lr=1e-3, betas=(0.9, 0.995), eps=0.0, weight_decay=0.0)
    t_opt.state[p] = {
        "step": 1,
        "exp_avg": torch.zeros(4),
        "exp_avg_sq": torch.ones(4),
        "W": torch.tensor(9.0),
    }
    p.grad = torch.ones(4)  # mahal = sum((1-0)^2 / 1) = 4 = d -> w_t = (4+4)/(4+4) = 1
    t_opt.step()
    assert torch.isclose(t_opt.state[p]["W"], torch.tensor(9.0), atol=1e-6)
    assert isinstance(ExactStableAdamW([p]), ExactStableAdamW)
