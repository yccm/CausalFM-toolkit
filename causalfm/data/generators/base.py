"""
Base classes for data generation using DAG-structured SCMs.

This module provides the foundational building blocks for generating
synthetic causal datasets based on MLP-like graph structures.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


class DAGStructuredSCM:
    """
    Implementation of DAG-structured Structural Causal Models (SCMs) sampler
    based on a modified MLP architecture.
    
    This class generates synthetic covariates by constructing an MLP-like
    directed acyclic graph and sampling from the structural equations.
    
    Example:
        >>> scm = DAGStructuredSCM(edge_drop_prob=0.4)
        >>> X = scm.generate_dataset(num_features=10, num_samples=1000)
        >>> print(X.shape)  # (1000, 10)
    """
    
    def __init__(
        self,
        prior_layers: Callable[[], int] = lambda: np.random.randint(2, 6),
        prior_hidden_size: Callable[[], int] = lambda: np.random.randint(10, 50),
        prior_weight: Callable[[], float] = lambda: np.random.normal(0, 1),
        edge_drop_prob: float = 0.4,
        activation: Callable[[np.ndarray], np.ndarray] = lambda x: np.tanh(x)
    ):
        """
        Initialize the DAG-structured SCM sampler.
        
        Args:
            prior_layers: Callable returning the number of layers
            prior_hidden_size: Callable returning the hidden size
            prior_weight: Callable returning edge weights
            edge_drop_prob: Probability of dropping an edge to create sparsity
            activation: Activation function to use (default: tanh)
        """
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.edge_drop_prob = edge_drop_prob
        self.activation = activation
        self.dag: Optional[nx.DiGraph] = None
        self.weights: Dict = {}
        self.biases: Dict = {}
        self.noise_distributions: Dict = {}
        self.feature_nodes: np.ndarray = np.array([])
        self.node_values: Dict = {}
        self.topological_order: list = []
        
    def sample_noise_distribution(self) -> Callable:
        """
        Sample a noise distribution from a meta-distribution.
        
        Returns:
            A callable that samples from the chosen noise distribution
        """
        dist_type = np.random.choice(["normal", "uniform", "laplace", "logistic"])
        
        if dist_type == "normal":
            scale = np.random.uniform(0.1, 2.0)
            return lambda size: np.random.normal(0, scale, size)
        elif dist_type == "uniform":
            scale = np.random.uniform(0.1, 2.0)
            return lambda size: np.random.uniform(-scale, scale, size)
        elif dist_type == "laplace":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.laplace(0, scale, size)
        else:  # logistic
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.logistic(0, scale, size)
    
    def construct_mlp_graph(self, num_layers: int, hidden_size: int) -> nx.DiGraph:
        """
        Construct an MLP-like directed graph structure.
        
        Args:
            num_layers: Number of layers in the MLP
            hidden_size: Size of hidden layers
            
        Returns:
            A directed graph representing the MLP structure
        """
        G = nx.DiGraph()
        node_id = 0
        layer_sizes = [hidden_size] * num_layers
        nodes_by_layer = []
        
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        
        for layer_idx in range(num_layers - 1):
            src_nodes = nodes_by_layer[layer_idx]
            dst_nodes = nodes_by_layer[layer_idx + 1]
            for src in src_nodes:
                for dst in dst_nodes:
                    G.add_edge(src, dst)
        
        return G
    
    def transform_to_dag(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Transform graph by randomly dropping edges.
        
        Args:
            G: Input directed graph
            
        Returns:
            DAG with some edges dropped
        """
        dag = G.copy()
        edges = list(G.edges())
        num_edges_to_drop = int(len(edges) * self.edge_drop_prob)
        
        if num_edges_to_drop > 0:
            edges_to_drop = np.random.choice(
                len(edges), size=num_edges_to_drop, replace=False
            )
            for idx in edges_to_drop:
                u, v = edges[idx]
                dag.remove_edge(u, v)
        
        assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG after edge removal"
        return dag
    
    def sample_structural_equation_parameters(self) -> None:
        """Sample weights, biases, and noise distributions for each node."""
        for node in self.dag.nodes():
            parents = list(self.dag.predecessors(node))
            if parents:
                self.weights[node] = {
                    parent: self.prior_weight() for parent in parents
                }
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()
    
    def evaluate_node(self, node: int) -> float:
        """
        Evaluate the value of a node given its parents.
        
        Args:
            node: Node ID to evaluate
            
        Returns:
            Computed value for the node
        """
        parents = list(self.dag.predecessors(node))
        
        if not parents:
            noise = self.noise_distributions[node](1)[0]
            value = self.activation(self.biases[node] + noise)
        else:
            weighted_sum = sum(
                self.weights[node][parent] * self.node_values[parent] 
                for parent in parents
            )
            noise = self.noise_distributions[node](1)[0]
            value = self.activation(weighted_sum + self.biases[node] + noise)
            
        return value
    
    def sample_observation(self) -> np.ndarray:
        """
        Sample one observation from the SCM.
        
        Returns:
            Array of feature values
        """
        self.node_values = {}
        for node in self.topological_order:
            self.node_values[node] = self.evaluate_node(node)
        return np.array([self.node_values[node] for node in self.feature_nodes])
    
    def generate_dataset(self, num_features: int, num_samples: int) -> np.ndarray:
        """
        Generate a dataset of samples from the SCM.
        
        Args:
            num_features: Number of features to generate
            num_samples: Number of samples to generate
            
        Returns:
            Array of shape (num_samples, num_features)
        """
        num_layers = self.prior_layers()
        hidden_size = self.prior_hidden_size()
        
        mlp_graph = self.construct_mlp_graph(num_layers, hidden_size)
        self.dag = self.transform_to_dag(mlp_graph)
        self.topological_order = list(nx.topological_sort(self.dag))
        self.sample_structural_equation_parameters()
        
        all_nodes = list(self.dag.nodes())
        assert num_features <= len(all_nodes), "Requested more features than available nodes"
        self.feature_nodes = np.random.choice(all_nodes, size=num_features, replace=False)
        
        dataset = np.zeros((num_samples, num_features))
        for i in range(num_samples):
            dataset[i] = self.sample_observation()
            
        return dataset


class BaseMLPGenerator:
    """
    Base class for MLP-like network generators.
    
    Provides common functionality for generating treatments, mediators,
    and outcomes using MLP-based structural equations.
    """
    
    def __init__(
        self,
        prior_layers: Callable[[], int] = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable[[], int] = lambda: np.random.randint(8, 30),
        prior_weight: Callable[[], float] = lambda: np.random.normal(0, 1),
        activation: Callable[[np.ndarray], np.ndarray] = lambda x: np.tanh(x),
        use_layer_noise: bool = True,
        layer_noise_scale: float = 0.1,
    ):
        """
        Initialize the MLP generator.
        
        Args:
            prior_layers: Callable returning the number of layers
            prior_hidden_size: Callable returning the hidden size
            prior_weight: Callable returning edge weights
            activation: Activation function (default: tanh)
            use_layer_noise: Whether to add noise at each layer
            layer_noise_scale: Scale of the layer noise
        """
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.activation = activation
        self.use_layer_noise = use_layer_noise
        self.layer_noise_scale = layer_noise_scale
        
        self.network: Optional[nx.DiGraph] = None
        self.weights: Dict = {}
        self.biases: Dict = {}
        self.noise_distributions: Dict = {}
        self.output_node: Optional[int] = None

    def sample_noise_distribution(self) -> Callable:
        """Sample a noise distribution from a meta-distribution."""
        dist_type = np.random.choice(["normal", "uniform", "laplace", "logistic"])
        scale_map = {
            "normal": (0.1, 1.0),
            "uniform": (0.1, 1.0),
            "laplace": (0.1, 0.5),
            "logistic": (0.1, 0.5)
        }
        scale = np.random.uniform(*scale_map[dist_type])
        
        dist_map = {
            "normal": lambda s: np.random.normal(0, scale, s),
            "uniform": lambda s: np.random.uniform(-scale, scale, s),
            "laplace": lambda s: np.random.laplace(0, scale, s),
            "logistic": lambda s: np.random.logistic(0, scale, s)
        }
        return dist_map[dist_type]

    def _construct_network(
        self, 
        num_layers: int, 
        hidden_size: int, 
        input_size: int, 
        output_size: int
    ) -> nx.DiGraph:
        """
        Construct an MLP-like network.
        
        Args:
            num_layers: Number of layers
            hidden_size: Size of hidden layers
            input_size: Number of input features
            output_size: Number of outputs
            
        Returns:
            Directed graph representing the network
        """
        G = nx.DiGraph()
        node_id = 0
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [output_size]
        nodes_by_layer = []
        
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = [node_id + i for i in range(size)]
            nodes_by_layer.append(layer_nodes)
            for node in layer_nodes:
                G.add_node(node, layer=layer_idx)
            node_id += size

        if output_size == 1:
            self.output_node = nodes_by_layer[-1][0]
        
        for i in range(num_layers - 1):
            for src in nodes_by_layer[i]:
                for dst in nodes_by_layer[i + 1]:
                    G.add_edge(src, dst)
                    
        return G

    def _sample_network_parameters(self) -> None:
        """Sample weights, biases, and noise distributions for the network."""
        for node in self.network.nodes():
            parents = list(self.network.predecessors(node))
            if parents:
                self.weights[node] = {parent: self.prior_weight() for parent in parents}
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()

    def _forward_propagate(self, z: np.ndarray) -> float:
        """
        Forward propagate input through the network.
        
        Args:
            z: Input array
            
        Returns:
            Output value
        """
        node_values = {}
        node_layers = nx.get_node_attributes(self.network, 'layer')
        
        for i in range(len(z)):
            node_values[i] = z[i]
            
        for layer in sorted(list(set(node_layers.values())))[1:]:
            for node in [n for n, l in node_layers.items() if l == layer]:
                parents = list(self.network.predecessors(node))
                weighted_sum = sum(
                    self.weights[node][p] * node_values[p] for p in parents
                )

                if self.use_layer_noise:
                    noise = np.random.normal(0, self.layer_noise_scale)
                else:
                    noise = 0.0
                
                value = weighted_sum + self.biases[node] + noise
                if node != self.output_node:
                    value = self.activation(value)
                node_values[node] = value
                
        return node_values[self.output_node]
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class BaseDataGenerator(ABC):
    """
    Abstract base class for all data generators.
    
    Provides a common interface for generating causal datasets.
    """
    
    def __init__(
        self,
        num_samples: int = 1024,
        num_features: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize the data generator.
        
        Args:
            num_samples: Number of samples per dataset
            num_features: Number of features (covariates)
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_features = num_features
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
    
    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """
        Generate a single dataset.
        
        Returns:
            DataFrame containing the generated data
        """
        pass
    
    @abstractmethod
    def generate_multiple(
        self, 
        num_datasets: int, 
        output_dir: str
    ) -> None:
        """
        Generate multiple datasets and save to files.
        
        Args:
            num_datasets: Number of datasets to generate
            output_dir: Directory to save the datasets
        """
        pass
    
    def save_dataset(
        self, 
        df: pd.DataFrame, 
        filepath: str
    ) -> None:
        """
        Save dataset to a CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the file
        """
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")


