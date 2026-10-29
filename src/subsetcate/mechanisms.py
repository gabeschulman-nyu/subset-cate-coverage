from collections.abc import Callable
from copy import deepcopy
from typing import AnyStr

import dowhy
import networkx as nx
import numpy as np
from dowhy import gcm
from scipy import stats


class GroundTruthPredictionModel(gcm.PredictionModel):
    def __init__(self, prediction_function: Callable[[np.ndarray], float | np.ndarray]) -> None:
        super().__init__()
        self.prediction_function = prediction_function

    def fit(self, X, Y):
        pass

    def predict(self, X):
        Y = self.prediction_function(X)
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        return Y

    def clone(self):
        return GroundTruthPredictionModel(prediction_function=deepcopy(self.prediction_function))


def build_synthetic_causal_model(
    causal_graph: nx.DiGraph,
    node_function_dict: dict[
        AnyStr,
        tuple[gcm.PredictionModel | Callable | None, gcm.ScipyDistribution],
    ],
) -> gcm.InvertibleStructuralCausalModel:
    causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)
    for node in causal_graph.nodes:
        noise_model = node_function_dict[node][1]

        if noise_model is None:
            noise_model = gcm.ScipyDistribution(stats.norm, loc=0, scale=0)

        if not noise_model._fixed_parameters:
            raise ValueError(f"Noise model for node {node} is not parameterized.")

        node_parents = list(causal_graph.predecessors(node))
        if len(node_parents) > 0:
            prediction_function = node_function_dict[node][0]
            if isinstance(prediction_function, gcm.PredictionModel):
                prediction_model = prediction_function
            else:
                prediction_model = GroundTruthPredictionModel(
                    prediction_function=prediction_function
                )
            current_mechanism = gcm.AdditiveNoiseModel(
                prediction_model=prediction_model, noise_model=noise_model
            )
        else:
            current_mechanism = noise_model

        causal_model.set_causal_mechanism(node, current_mechanism)

        causal_model.graph.nodes[node]["parents_during_fit"] = dowhy.graph.get_ordered_predecessors(
            causal_graph=causal_model.graph, node=node
        )

    return causal_model


