from __future__ import annotations

from src.domain.models import AlgorithmSpec, ParameterSchema


# Conjunto de testes de independência condicional suportados pela causal-learn
# para algoritmos baseados em restrições (PC, FCI). Cada teste é apropriado
# para um tipo de dado diferente:
#   - "fisherz":     ótimo para dados Gaussianos contínuos.
#   - "chisq":       para variáveis puramente categóricas (e.g. dados
#                    discretizados em bins, como em Ramos et al. 2024).
#   - "gsq":         G-squared (likelihood ratio); alternativa ao chisq.
#   - "kci":         kernel-based, captura dependências não-lineares; cúbico
#                    no tamanho da amostra, lento em datasets grandes.
#   - "mv_fisherz":  variante do Fisher-Z tolerante a valores faltantes.
#
# Atenção: o backend py-tetrad NÃO aplica automaticamente este parâmetro
# (usa seu próprio default interno). Para escolha do teste com fidelidade,
# o fallback causal-learn é o caminho garantido. Esta lista cobre o que o
# causal-learn aceita; ver
# https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/PC.html
_INDEP_TEST_OPTIONS = ["fisherz", "chisq", "gsq", "kci", "mv_fisherz"]


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
            ParameterSchema("indep_test", "Independence test", "select", "fisherz", options=_INDEP_TEST_OPTIONS),
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
            ParameterSchema("indep_test", "Independence test", "select", "fisherz", options=_INDEP_TEST_OPTIONS),
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
        # Default reduzido de 100 → 30 simulações. Com 30, p-value tem precisão
        # ~5% (suficiente para veredicto qualitativo de robustez), e a
        # alocação de memória cai 70%. Importante para evitar SIGSEGV em
        # libarrow.so quando muitos pares são avaliados em sequência.
        ParameterSchema("num_simulations", "Number of simulations", "int", 30, min_value=1, max_value=5000, step=1),
    ],
    "random_common_cause": [
        ParameterSchema("num_simulations", "Number of simulations", "int", 30, min_value=1, max_value=5000, step=1),
    ],
    "data_subset_refuter": [
        ParameterSchema("subset_fraction", "Subset fraction", "float", 0.8, min_value=0.1, max_value=0.99, step=0.01),
        ParameterSchema("num_simulations", "Number of simulations", "int", 30, min_value=1, max_value=5000, step=1),
    ],
    "add_unobserved_common_cause": [
        # Estes parâmetros DEVEM bater com a API do DoWhy. Vide
        # https://www.pywhy.org/dowhy/.../add_unobserved_common_cause:
        #   - confounders_effect_on_* é STRING ("linear" ou "binary_flip"),
        #     descreve a forma do efeito (regressão para contínuo, flip para binário).
        #   - effect_strength_on_* é FLOAT, parâmetro da intensidade (coeficiente
        #     de regressão para "linear", probabilidade de flip para "binary_flip").
        ParameterSchema(
            "confounders_effect_on_treatment", "Effect form on treatment",
            "select", "linear", options=["linear", "binary_flip"],
        ),
        ParameterSchema(
            "confounders_effect_on_outcome", "Effect form on outcome",
            "select", "linear", options=["linear", "binary_flip"],
        ),
        ParameterSchema(
            "effect_strength_on_treatment", "Effect strength on treatment",
            "float", 0.1, min_value=0.0, max_value=10.0, step=0.01,
        ),
        ParameterSchema(
            "effect_strength_on_outcome", "Effect strength on outcome",
            "float", 0.1, min_value=0.0, max_value=10.0, step=0.01,
        ),
    ],
}