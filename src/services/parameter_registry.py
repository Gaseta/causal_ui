from __future__ import annotations

from src.domain.models import AlgorithmSpec, ParameterSchema


DISCOVERY_ALGORITHMS = [
    AlgorithmSpec(
        key="pc",
        label="PC",
        family="Constraint-based",
        backend="pytetrad_or_causallearn",
        runner="pc",
        description="Constraint-based search for approximately Gaussian data.",
        parameters=[
            ParameterSchema("alpha", "Alpha", "float", 0.05, min_value=0.0001, max_value=1.0, step=0.001),
            ParameterSchema("stable", "Stable", "bool", True),
            ParameterSchema("max_k", "Max K", "int", 3, min_value=1, max_value=10, step=1),
            ParameterSchema("indep_test", "Independence test", "select", "fisherz", options=["fisherz"]),
        ],
    ),
    AlgorithmSpec(
        key="fges",
        label="FGES",
        family="Score-based",
        backend="pytetrad_or_causallearn",
        runner="fges",
        description="Fast Greedy Equivalence Search.",
        parameters=[
            ParameterSchema("score", "Score", "select", "sem_bic", options=["sem_bic"]),
            ParameterSchema("penalty_discount", "Penalty discount", "float", 2.0, min_value=0.1, max_value=20.0, step=0.1),
        ],
    ),
    AlgorithmSpec(
        key="ges",
        label="GES",
        family="Score-based",
        backend="causallearn",
        runner="ges",
        description="Greedy Equivalence Search via causal-learn fallback.",
        parameters=[],
    ),
    AlgorithmSpec(
        key="fci",
        label="FCI",
        family="Constraint-based",
        backend="causallearn",
        runner="fci",
        description="Fast Causal Inference for latent confounding scenarios.",
        parameters=[
            ParameterSchema("alpha", "Alpha", "float", 0.05, min_value=0.0001, max_value=1.0, step=0.001),
            ParameterSchema("indep_test", "Independence test", "select", "fisherz", options=["fisherz"]),
        ],
    ),
    AlgorithmSpec(
        key="gfci",
        label="GFCI",
        family="Hybrid",
        backend="placeholder",
        runner="gfci",
        description="Prepared for py-tetrad extension point. Requires environment-specific bridge implementation.",
        parameters=[
            ParameterSchema("alpha", "Alpha", "float", 0.05, min_value=0.0001, max_value=1.0, step=0.001),
            ParameterSchema("penalty_discount", "Penalty discount", "float", 2.0, min_value=0.1, max_value=20.0, step=0.1),
        ],
    ),
]


ESTIMATE_METHODS = {
    "backdoor": [
        "backdoor.linear_regression",
        "backdoor.generalized_linear_model",
        "backdoor.propensity_score_matching",
        "backdoor.propensity_score_stratification",
        "backdoor.propensity_score_weighting",
        "backdoor.econml.dml.DML",
        "backdoor.econml.dr.LinearDRLearner",
    ],
    "iv": [
        "iv.instrumental_variable",
    ],
}


ESTIMATE_METHOD_SPECS = {
    "backdoor.linear_regression": [
        ParameterSchema("test_significance", "Test significance", "bool", False),
        ParameterSchema("confidence_intervals", "Confidence intervals", "bool", False),
    ],
    "backdoor.generalized_linear_model": [
        ParameterSchema("glm_family", "GLM family", "select", "gaussian", options=["gaussian", "binomial", "poisson"]),
        ParameterSchema("test_significance", "Test significance", "bool", False),
        ParameterSchema("confidence_intervals", "Confidence intervals", "bool", False),
    ],
    "backdoor.propensity_score_matching": [
        ParameterSchema("target_units", "Target units", "select", "ate", options=["ate", "att", "atc"]),
    ],
    "backdoor.propensity_score_stratification": [
        ParameterSchema("num_strata", "Number of strata", "int", 50, min_value=2, max_value=500, step=1),
        ParameterSchema("clipping_threshold", "Clipping threshold", "int", 10, min_value=1, max_value=1000, step=1),
        ParameterSchema("target_units", "Target units", "select", "ate", options=["ate", "att", "atc"]),
    ],
    "backdoor.propensity_score_weighting": [
        ParameterSchema("min_ps_score", "Min propensity score", "float", 0.05, min_value=0.0, max_value=0.5, step=0.01),
        ParameterSchema("max_ps_score", "Max propensity score", "float", 0.95, min_value=0.5, max_value=1.0, step=0.01),
        ParameterSchema("weighting_scheme", "Weighting scheme", "select", "ips_weight", options=["ips_weight", "ips_stabilized_weight", "ips_normalized_weight"]),
        ParameterSchema("target_units", "Target units", "select", "ate", options=["ate", "att", "atc"]),
    ],
    "backdoor.econml.dml.DML": [
        ParameterSchema("discrete_treatment", "Discrete treatment", "bool", False),
        ParameterSchema("cv", "Cross-validation folds", "int", 3, min_value=2, max_value=20, step=1),
    ],
    "backdoor.econml.dr.LinearDRLearner": [
        ParameterSchema("cv", "Cross-validation folds", "int", 3, min_value=2, max_value=20, step=1),
    ],
    "iv.instrumental_variable": [],
}


REFUTERS = [
    "placebo_treatment_refuter",
    "random_common_cause",
    "data_subset_refuter",
    "add_unobserved_common_cause",
]


REFUTER_SPECS = {
    "placebo_treatment_refuter": [
        ParameterSchema("placebo_type", "Placebo type", "select", "permute", options=["permute", "Random Data"]),
        ParameterSchema("num_simulations", "Number of simulations", "int", 100, min_value=1, max_value=5000, step=1),
    ],
    "random_common_cause": [
        ParameterSchema("num_simulations", "Number of simulations", "int", 100, min_value=1, max_value=5000, step=1),
    ],
    "data_subset_refuter": [
        ParameterSchema("subset_fraction", "Subset fraction", "float", 0.8, min_value=0.1, max_value=0.99, step=0.01),
        ParameterSchema("num_simulations", "Number of simulations", "int", 100, min_value=1, max_value=5000, step=1),
    ],
    "add_unobserved_common_cause": [
        ParameterSchema("confounders_effect_on_treatment", "Effect on treatment", "float", 0.1, min_value=0.0, max_value=10.0, step=0.1),
        ParameterSchema("confounders_effect_on_outcome", "Effect on outcome", "float", 0.1, min_value=0.0, max_value=10.0, step=0.1),
    ],
}
