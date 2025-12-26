"""
Front-door adjustment data generator.

This module provides functionality for generating synthetic datasets
for causal inference using the front-door criterion.
"""

import os
from typing import Callable, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from causalfm.data.generators.base import BaseDataGenerator, DAGStructuredSCM


class TreatmentAssignerWithConfounding:
    """
    Treatment assigner that incorporates unobserved confounding.
    
    Treatment depends on both observed covariates X and unobserved confounders U.
    """
    
    def __init__(
        self,
        prior_layers: Callable[[], int] = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable[[], int] = lambda: np.random.randint(8, 30),
        prior_weight: Callable[[], float] = lambda: np.random.normal(0, 1),
        activation: Callable = lambda x: np.tanh(x)
    ):
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.activation = activation
        self.treatment_network: Optional[nx.DiGraph] = None
        self.weights: Dict = {}
        self.biases: Dict = {}
        self.output_node: Optional[int] = None
        self.propensity_scores: Optional[np.ndarray] = None

    def construct_treatment_network(
        self, num_layers: int, hidden_size: int, input_size: int
    ) -> nx.DiGraph:
        """Construct the treatment network."""
        G = nx.DiGraph()
        node_id = 0
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [1]
        nodes_by_layer = []
        
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)

        self.output_node = nodes_by_layer[-1][0]

        for layer_idx in range(num_layers - 1):
            for src in nodes_by_layer[layer_idx]:
                for dst in nodes_by_layer[layer_idx + 1]:
                    G.add_edge(src, dst)
        return G

    def sample_treatment_network_parameters(self) -> None:
        """Sample network parameters."""
        for node in self.treatment_network.nodes():
            parents = list(self.treatment_network.predecessors(node))
            if parents:
                self.weights[node] = {
                    parent: self.prior_weight() for parent in parents
                }
            self.biases[node] = self.prior_weight()

    def forward_propagate(self, xu: np.ndarray) -> Dict[int, float]:
        """Forward propagate through the network."""
        node_values = {}
        node_layers = nx.get_node_attributes(self.treatment_network, 'layer')
        max_layer = max(node_layers.values())

        for layer in range(max_layer + 1):
            layer_nodes = [n for n, l in node_layers.items() if l == layer]
            for node in layer_nodes:
                parents = list(self.treatment_network.predecessors(node))
                if layer == 0:
                    node_values[node] = xu[node] if node < len(xu) else 0.0
                else:
                    weighted_sum = sum(
                        self.weights[node][p] * node_values[p] for p in parents
                    )
                    value = weighted_sum + self.biases[node]
                    if node != self.output_node:
                        value = self.activation(value)
                    node_values[node] = value
        return node_values

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def generate_treatments(
        self, X: np.ndarray, U: np.ndarray
    ) -> np.ndarray:
        """
        Generate treatments based on X and U.
        
        Args:
            X: Observed covariates
            U: Unobserved confounders
            
        Returns:
            Array of binary treatments
        """
        n_samples = X.shape[0]
        n_features_total = X.shape[1] + U.shape[1]

        num_layers = max(3, self.prior_layers())
        hidden_size = self.prior_hidden_size()

        self.treatment_network = self.construct_treatment_network(
            num_layers=num_layers,
            hidden_size=hidden_size,
            input_size=n_features_total
        )
        self.sample_treatment_network_parameters()

        treatments = np.zeros(n_samples, dtype=int)
        propensity_scores = np.zeros(n_samples, dtype=float)

        for i in range(n_samples):
            xu = np.concatenate([X[i], U[i]])
            node_values = self.forward_propagate(xu)
            logit = node_values[self.output_node]
            propensity = self.sigmoid(logit)
            propensity_scores[i] = propensity
            treatments[i] = np.random.binomial(1, propensity)

        self.propensity_scores = propensity_scores
        return treatments


class MediatorGenerator:
    """
    Generates mediator M from covariates X and treatment A.
    
    The mediator is on the causal path from A to Y and blocks
    the backdoor path through unobserved confounders.
    """
    
    def __init__(
        self,
        prior_layers: Callable[[], int] = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable[[], int] = lambda: np.random.randint(8, 30),
        prior_weight: Callable[[], float] = lambda: np.random.normal(0, 1),
        activation: Callable = lambda x: np.tanh(x)
    ):
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.activation = activation
        self.mediator_network: Optional[nx.DiGraph] = None
        self.weights: Dict = {}
        self.biases: Dict = {}
        self.noise_distributions: Dict = {}
        self.output_node: Optional[int] = None
        
    def sample_noise_distribution(self) -> Callable:
        """Sample a noise distribution."""
        dist_type = np.random.choice(["normal", "uniform", "laplace", "logistic"])
        
        if dist_type == "normal":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.normal(0, scale, size)
        elif dist_type == "uniform":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.uniform(-scale, scale, size)
        elif dist_type == "laplace":
            scale = np.random.uniform(0.1, 0.5)
            return lambda size: np.random.laplace(0, scale, size)
        else:
            scale = np.random.uniform(0.1, 0.5)
            return lambda size: np.random.logistic(0, scale, size)
    
    def construct_mediator_network(
        self, num_layers: int, hidden_size: int, input_size: int
    ) -> nx.DiGraph:
        """Construct the mediator network."""
        G = nx.DiGraph()
        node_id = 0
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [1]
        nodes_by_layer = []
        
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        
        self.output_node = nodes_by_layer[-1][0]
        
        for layer_idx in range(num_layers - 1):
            for src in nodes_by_layer[layer_idx]:
                for dst in nodes_by_layer[layer_idx + 1]:
                    G.add_edge(src, dst)
        
        return G
    
    def sample_mediator_network_parameters(self) -> None:
        """Sample network parameters."""
        for node in self.mediator_network.nodes():
            parents = list(self.mediator_network.predecessors(node))
            if parents:
                self.weights[node] = {
                    parent: self.prior_weight() for parent in parents
                }
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()
    
    def forward_propagate(
        self, xa: np.ndarray, noise_samples: Dict[int, float]
    ) -> Dict[int, float]:
        """Forward propagate through the network."""
        node_values = {}
        node_layers = nx.get_node_attributes(self.mediator_network, 'layer')
        max_layer = max(node_layers.values())
        
        for layer in range(max_layer + 1):
            layer_nodes = [n for n, l in node_layers.items() if l == layer]
            for node in layer_nodes:
                parents = list(self.mediator_network.predecessors(node))
                
                if layer == 0:
                    node_values[node] = xa[node] if node < len(xa) else 0
                else:
                    weighted_sum = sum(
                        self.weights[node][parent] * node_values[parent] 
                        for parent in parents
                    )
                    noise = noise_samples[node]
                    value = weighted_sum + self.biases[node] + noise
                    if node != self.output_node:
                        value = self.activation(value)
                    node_values[node] = value
        
        return node_values
    
    def generate_mediators(
        self, X: np.ndarray, A: np.ndarray
    ) -> np.ndarray:
        """
        Generate mediator values.
        
        Args:
            X: Covariates
            A: Treatment values
            
        Returns:
            Array of mediator values
        """
        n_samples = X.shape[0]
        n_features_total = X.shape[1] + 1
        
        num_layers = max(3, self.prior_layers())
        hidden_size = self.prior_hidden_size()
        
        self.mediator_network = self.construct_mediator_network(
            num_layers, hidden_size, n_features_total
        )
        self.sample_mediator_network_parameters()
        
        mediators = np.zeros(n_samples)
        
        for i in range(n_samples):
            xa = np.concatenate([X[i], [A[i]]])
            noise_samples = {
                node: self.noise_distributions[node](1)[0]
                for node in self.mediator_network.nodes()
            }
            node_values = self.forward_propagate(xa, noise_samples)
            mediators[i] = node_values[self.output_node]
        
        return mediators


class OutcomeGeneratorWithMediator:
    """
    Outcome generator for front-door setting.
    
    Outcome depends on X, U, and M (mediator).
    """
    
    def __init__(
        self,
        prior_layers: Callable[[], int] = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable[[], int] = lambda: np.random.randint(8, 30),
        prior_weight: Callable[[], float] = lambda: np.random.normal(0, 1),
        edge_drop_prob: float = 0.3,
        activation: Callable = lambda x: np.tanh(x)
    ):
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.edge_drop_prob = edge_drop_prob
        self.activation = activation
        self.outcome_network: Optional[nx.DiGraph] = None
        self.weights: Dict = {}
        self.biases: Dict = {}
        self.noise_distributions: Dict = {}
        self.output_node: Optional[int] = None
        
    def sample_noise_distribution(self) -> Callable:
        """Sample a noise distribution."""
        dist_type = np.random.choice(["normal", "uniform", "laplace", "logistic"])
        
        if dist_type == "normal":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.normal(0, scale, size)
        elif dist_type == "uniform":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.uniform(-scale, scale, size)
        elif dist_type == "laplace":
            scale = np.random.uniform(0.1, 0.5)
            return lambda size: np.random.laplace(0, scale, size)
        else:
            scale = np.random.uniform(0.1, 0.5)
            return lambda size: np.random.logistic(0, scale, size)
    
    def construct_outcome_network(
        self, num_layers: int, hidden_size: int, input_size: int
    ) -> nx.DiGraph:
        """Construct the outcome network."""
        G = nx.DiGraph()
        node_id = 0
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [1]
        nodes_by_layer = []
        
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        
        self.output_node = nodes_by_layer[-1][0]
        
        for layer_idx in range(num_layers - 1):
            for src in nodes_by_layer[layer_idx]:
                for dst in nodes_by_layer[layer_idx + 1]:
                    G.add_edge(src, dst)
        
        # Drop edges
        edges = list(G.edges())
        num_edges_to_drop = int(len(edges) * self.edge_drop_prob)
        if num_edges_to_drop > 0:
            edges_to_drop = np.random.choice(
                len(edges), size=num_edges_to_drop, replace=False
            )
            for idx in edges_to_drop:
                u, v = edges[idx]
                if v != self.output_node:
                    G.remove_edge(u, v)
        
        return G
    
    def sample_outcome_network_parameters(self) -> None:
        """Sample network parameters."""
        for node in self.outcome_network.nodes():
            parents = list(self.outcome_network.predecessors(node))
            if parents:
                self.weights[node] = {
                    parent: self.prior_weight() for parent in parents
                }
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()
    
    def forward_propagate(
        self, xum: np.ndarray, noise_samples: Dict[int, float]
    ) -> Dict[int, float]:
        """Forward propagate through the network."""
        node_values = {}
        node_layers = nx.get_node_attributes(self.outcome_network, 'layer')
        max_layer = max(node_layers.values())
        
        for layer in range(max_layer + 1):
            layer_nodes = [n for n, l in node_layers.items() if l == layer]
            for node in layer_nodes:
                parents = list(self.outcome_network.predecessors(node))
                
                if layer == 0:
                    node_values[node] = xum[node] if node < len(xum) else 0
                else:
                    weighted_sum = sum(
                        self.weights[node][parent] * node_values[parent] 
                        for parent in parents
                    ) if parents else 0
                    noise = noise_samples[node]
                    value = weighted_sum + self.biases[node] + noise
                    if node != self.output_node:
                        value = self.activation(value)
                    node_values[node] = value
        
        return node_values
    
    def generate_outcomes(
        self, 
        X: np.ndarray, 
        U: np.ndarray,
        M: np.ndarray,
        A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate outcomes for front-door setting.
        
        Args:
            X: Observed covariates
            U: Unobserved confounders
            M: Mediator values
            A: Treatment values
            
        Returns:
            Tuple of (Y, Y0, Y1, M0, M1)
        """
        n_samples = X.shape[0]
        n_features_total = X.shape[1] + U.shape[1] + 1
        
        num_layers = max(3, self.prior_layers())
        hidden_size = self.prior_hidden_size()
        
        self.outcome_network = self.construct_outcome_network(
            num_layers, hidden_size, n_features_total
        )
        self.sample_outcome_network_parameters()
        
        # Generate mediators for counterfactual treatments
        mediator_gen = MediatorGenerator(
            prior_layers=lambda: np.random.randint(2, 5),
            prior_hidden_size=lambda: np.random.randint(8, 30),
            prior_weight=lambda: np.random.normal(0, 1),
            activation=lambda x: np.tanh(x)
        )
        
        A0 = np.zeros(n_samples, dtype=int)
        A1 = np.ones(n_samples, dtype=int)
        M0 = mediator_gen.generate_mediators(X, A0)
        M1 = mediator_gen.generate_mediators(X, A1)
        
        Y = np.zeros(n_samples, dtype=float)
        Y0 = np.zeros(n_samples, dtype=float)
        Y1 = np.zeros(n_samples, dtype=float)
        
        for i in range(n_samples):
            noise_samples = {
                node: self.noise_distributions[node](1)[0]
                for node in self.outcome_network.nodes()
            }
            
            # Factual outcome
            xum = np.concatenate([X[i], U[i], [M[i]]])
            node_values = self.forward_propagate(xum, noise_samples)
            Y[i] = node_values[self.output_node]
            
            # Counterfactual outcomes
            xum0 = np.concatenate([X[i], U[i], [M0[i]]])
            node_values0 = self.forward_propagate(xum0, noise_samples)
            Y0[i] = node_values0[self.output_node]
            
            xum1 = np.concatenate([X[i], U[i], [M1[i]]])
            node_values1 = self.forward_propagate(xum1, noise_samples)
            Y1[i] = node_values1[self.output_node]
        
        return Y, Y0, Y1, M0, M1


class FrontdoorDataGenerator(BaseDataGenerator):
    """
    Generator for Front-door adjustment datasets.
    
    Generates synthetic datasets for causal inference using the front-door
    criterion, where a mediator M blocks the backdoor path between treatment
    A and outcome Y through unobserved confounders U.
    
    Example:
        >>> generator = FrontdoorDataGenerator(
        ...     num_samples=1024,
        ...     num_features=10,
        ...     num_confounders=5,
        ...     seed=42
        ... )
        >>> df = generator.generate()
        >>> print(df.columns)
        Index(['x0', ..., 'u0', ..., 'treatment', 'mediator', 'outcome', ...], dtype='object')
    """
    
    def __init__(
        self,
        num_samples: int = 1024,
        num_features: int = 10,
        num_confounders: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the Front-door data generator.
        
        Args:
            num_samples: Number of samples per dataset
            num_features: Number of observed covariates
            num_confounders: Number of unobserved confounders (if None, randomly sampled)
            seed: Random seed for reproducibility
        """
        super().__init__(num_samples, num_features, seed)
        self.num_confounders = num_confounders
    
    def generate(self) -> pd.DataFrame:
        """
        Generate a single Front-door dataset.
        
        Returns:
            DataFrame with columns for X, U, A, M, Y, Y0, Y1, M0, M1, ITE
        """
        # Determine number of confounders
        if self.num_confounders is None:
            num_confounders = int(np.random.randint(1, 16))
        else:
            num_confounders = self.num_confounders
        
        # Generate observed covariates
        dag_scm_x = DAGStructuredSCM(
            prior_layers=lambda: np.random.randint(3, 7),
            prior_hidden_size=lambda: np.random.randint(15, 40),
            prior_weight=lambda: np.random.normal(0, 1),
            edge_drop_prob=0.5,
            activation=lambda x: np.tanh(x)
        )
        X = dag_scm_x.generate_dataset(self.num_features, self.num_samples)
        
        # Generate unobserved confounders
        dag_scm_u = DAGStructuredSCM(
            prior_layers=lambda: np.random.randint(3, 6),
            prior_hidden_size=lambda: np.random.randint(10, 30),
            prior_weight=lambda: np.random.normal(0, 1),
            edge_drop_prob=0.4,
            activation=lambda x: np.tanh(x)
        )
        U = dag_scm_u.generate_dataset(num_confounders, self.num_samples)
        
        # Generate treatment
        treatment_assigner = TreatmentAssignerWithConfounding(
            prior_layers=lambda: np.random.randint(3, 5),
            prior_hidden_size=lambda: np.random.randint(8, 20),
            prior_weight=lambda: np.random.normal(0, 0.8),
            activation=lambda x: np.tanh(x)
        )
        A = treatment_assigner.generate_treatments(X, U)
        
        # Generate mediator
        mediator_generator = MediatorGenerator(
            prior_layers=lambda: np.random.randint(3, 5),
            prior_hidden_size=lambda: np.random.randint(10, 25),
            prior_weight=lambda: np.random.normal(0, 1.0),
            activation=lambda x: np.tanh(x)
        )
        M = mediator_generator.generate_mediators(X, A)
        
        # Generate outcomes
        outcome_generator = OutcomeGeneratorWithMediator(
            prior_layers=lambda: np.random.randint(3, 6),
            prior_hidden_size=lambda: np.random.randint(10, 25),
            prior_weight=lambda: np.random.normal(0, 1.0),
            edge_drop_prob=0.4,
            activation=lambda x: np.tanh(x)
        )
        Y, Y0, Y1, M0, M1 = outcome_generator.generate_outcomes(X, U, M, A)
        
        # Create DataFrame
        feature_names_x = [f"x{i}" for i in range(self.num_features)]
        feature_names_u = [f"u{i}" for i in range(num_confounders)]
        
        df = pd.DataFrame(X, columns=feature_names_x)
        df_u = pd.DataFrame(U, columns=feature_names_u)
        df = pd.concat([df, df_u], axis=1)
        
        df['treatment'] = A
        df['mediator'] = M
        df['outcome'] = Y
        df['y0'] = Y0
        df['y1'] = Y1
        df['m0'] = M0
        df['m1'] = M1
        df['ite'] = Y1 - Y0
        df['ate'] = np.mean(Y1 - Y0)
        
        return df
    
    def generate_multiple(
        self,
        num_datasets: int,
        output_dir: str,
        filename_prefix: str = "synthetic_frontdoor_dataset"
    ) -> None:
        """
        Generate multiple Front-door datasets and save to files.
        
        Args:
            num_datasets: Number of datasets to generate
            output_dir: Directory to save the datasets
            filename_prefix: Prefix for the dataset filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {num_datasets} Front-door datasets...")
        
        for i in range(1, num_datasets + 1):
            if self.seed is not None:
                np.random.seed(self.seed + i)
            
            try:
                df = self.generate()
                filepath = os.path.join(output_dir, f"{filename_prefix}_{i}.csv")
                self.save_dataset(df, filepath)
                print(f"Generated dataset {i}/{num_datasets}")
            except Exception as e:
                print(f"Error generating dataset {i}: {str(e)}")
                continue
        
        print(f"Generation complete. Saved to {output_dir}")

