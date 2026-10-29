"""Experiment setup for the synthetic polynomial family, with exact targets.

All synthetic-family models train on this prior. Targets are closed-form, from
subsetcate.exact. Query names follow upstream conventions (train.py derives
predict_ate/query_x_type from them): PATE, STATE and GATE are aggregate
decodes; CATE and ITE are per-query decodes.
"""

from subsetcate.exact import (
    covariate_names,
    make_exact_cate_query_fn,
    make_exact_ite_query_fn,
    make_exact_pate_query_fn,
    make_exact_state_query_fn,
    make_polynomial_prior_fn,
)
from subsetcate.mixture import CausalMixtureDistribution


def add_args(parser):
    parser.add_argument("--num_covariates", type=int, default=3)
    parser.add_argument("--outcome_noise_scale", type=float, default=0.25)
    parser.add_argument("--lambda_scale", type=float, default=1.0)
    # DGP settings of the polynomial family; the defaults give the base family.
    parser.add_argument("--effect_scale", type=float, default=0.35)
    parser.add_argument("--unobs_noise_coef", type=float, default=0.0)
    parser.add_argument("--unobs_conf_coef", type=float, default=0.0)
    parser.add_argument("--overlap_floor", type=float, default=0.05)
    parser.add_argument("--overlap_sharpness", type=float, default=1.0)
    return parser


def build_mixture(scm, config=None):
    mix = CausalMixtureDistribution(causal_model=scm)
    mix.add_observational(weight=1.0)
    return mix


def _build_setup(config, prior_fn, experiment_name):
    d = config["num_covariates"]
    x_names = covariate_names(d)
    observed_covariates = ["y", "t"] + x_names

    queries = {
        "PATE": make_exact_pate_query_fn(),
        "STATE": make_exact_state_query_fn(outcome_name="y", treatment_name="t"),
        "GATE": make_exact_state_query_fn(outcome_name="y", treatment_name="t"),
        "CATE": make_exact_cate_query_fn(query_feature_names=observed_covariates),
        "ITE": make_exact_ite_query_fn(outcome_name="y", treatment_name="t"),
    }

    return {
        "experiment_name": experiment_name,
        "observed_covariates": observed_covariates,
        "outcome_name": "y",
        "treatment_name": "t",
        "causal_model_prior_fn": prior_fn,
        "mixture_builder_fn": build_mixture,
        "queries": queries,
        "source_names": ["Observational"],
        "empirical_max_len": None,
        "data_generation_keys": [
            "num_covariates",
            "outcome_noise_scale",
            "lambda_scale",
            "effect_scale",
            "unobs_noise_coef",
            "unobs_conf_coef",
            "overlap_floor",
            "overlap_sharpness",
        ],
    }


def _build_polynomial_setup(config):
    d = config["num_covariates"]
    noise_scale = config["outcome_noise_scale"]
    knobs = dict(
        effect_scale=config.get("effect_scale", 0.35),
        unobs_noise_coef=config.get("unobs_noise_coef", 0.0),
        unobs_conf_coef=config.get("unobs_conf_coef", 0.0),
        overlap_floor=config.get("overlap_floor", 0.05),
        overlap_sharpness=config.get("overlap_sharpness", 1.0),
    )
    x_names = covariate_names(d)
    observed_covariates = ["y", "t"] + x_names

    queries = {
        "PATE": make_exact_pate_query_fn(),
        "STATE": make_exact_state_query_fn(outcome_name="y", treatment_name="t"),
        # GATE is the STATE query applied to the subset's rows.
        "GATE": make_exact_state_query_fn(outcome_name="y", treatment_name="t"),
        "CATE": make_exact_cate_query_fn(query_feature_names=observed_covariates),
        "ITE": make_exact_ite_query_fn(outcome_name="y", treatment_name="t"),
    }

    return {
        "experiment_name": "polynomial-confounder",
        "observed_covariates": observed_covariates,
        "outcome_name": "y",
        "treatment_name": "t",
        "causal_model_prior_fn": make_polynomial_prior_fn(
            num_covariates=d, outcome_noise_scale=noise_scale, **knobs
        ),
        "mixture_builder_fn": build_mixture,
        "queries": queries,
        "source_names": ["Observational"],
        "empirical_max_len": None,
        "data_generation_keys": ["num_covariates", "outcome_noise_scale"],
    }


experiment_setup = {
    "polynomial-confounder": _build_polynomial_setup,
}
