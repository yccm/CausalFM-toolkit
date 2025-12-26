"""
Instrumental Variables (IV) data generator.

This module provides functionality for generating synthetic datasets
for causal inference with instrumental variables.
"""

import os
from typing import Callable, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from causalfm.data.generators.base import (
    BaseDataGenerator,
    BaseMLPGenerator,
    DAGStructuredSCM,
)


class InstrumentGenerator(BaseMLPGenerator):
    """
    Generates instrument Z from covariates X.
    
    The instrument can be binary or continuous.
    """
    
    def generate_instrument(
        self, 
        X: np.ndarray, 
        instrument_type: str = 'binary'
    ) -> np.ndarray:
        """
        Generate instrument values.
        
        Args:
            X: Covariate matrix of shape (n_samples, n_features)
            instrument_type: 'binary' or 'continuous'
            
        Returns:
            Array of instrument values
        """
        n_samples, n_features = X.shape
        num_layers = max(3, self.prior_layers())
        hidden_size = self.prior_hidden_size()
        
        self.network = self._construct_network(num_layers, hidden_size, n_features, 1)
        self._sample_network_parameters()
        
        Z = np.zeros(n_samples)
        for i in range(n_samples):
            logit = self._forward_propagate(X[i])
            if instrument_type == 'binary':
                prob = self.sigmoid(logit)
                Z[i] = np.random.binomial(1, prob)
            else:
                Z[i] = logit
        
        return Z


class IVTreatmentAssigner(BaseMLPGenerator):
    """
    Treatment assigner for IV setting.
    
    Treatment depends on covariates X, instrument Z, and unobserved confounders U.
    """
    
    def __init__(
        self,
        prior_layers: Callable[[], int] = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable[[], int] = lambda: np.random.randint(8, 30),
        prior_weight: Callable[[], float] = lambda: np.random.normal(0, 1),
        activation: Callable = lambda x: np.tanh(x),
    ):
        super().__init__(
            prior_layers=prior_layers,
            prior_hidden_size=prior_hidden_size,
            prior_weight=prior_weight,
            activation=activation,
            use_layer_noise=False,
        )
        self.propensity_scores: Optional[np.ndarray] = None

    def generate_treatments(
        self, 
        X: np.ndarray, 
        Z: np.ndarray, 
        U: np.ndarray
    ) -> np.ndarray:
        """
        Generate treatments based on X, Z, and U.
        
        Args:
            X: Observed covariates
            Z: Instrument values
            U: Unobserved confounders
            
        Returns:
            Array of binary treatments
        """
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        full_input = np.hstack([X, Z, U])
        n_samples, n_features_full = full_input.shape
        
        num_layers = max(3, self.prior_layers())
        hidden_size = self.prior_hidden_size()
        
        self.network = self._construct_network(num_layers, hidden_size, n_features_full, 1)
        self._sample_network_parameters()
        
        A = np.zeros(n_samples, dtype=int)
        self.propensity_scores = np.zeros(n_samples)

        for i in range(n_samples):
            logit = self._forward_propagate(full_input[i])
            propensity = self.sigmoid(logit)
            self.propensity_scores[i] = propensity
            A[i] = np.random.binomial(1, propensity)
            
        return A


class IVOutcomeGenerator:
    """
    Outcome generator for IV setting.
    
    Outcome depends on X, A, and U (unobserved confounders).
    Uses separate networks for treatment effect and confounding effect.
    """
    
    def __init__(
        self,
        prior_layers: Callable[[], int] = lambda: np.random.randint(3, 6),
        prior_hidden_size: Callable[[], int] = lambda: np.random.randint(10, 25),
        prior_weight: Callable[[], float] = lambda: np.random.normal(0, 1.0),
        activation: Callable = lambda x: np.tanh(x),
        outcome_noise_scale: float = 0.5
    ):
        """
        Initialize the IV outcome generator.
        
        Args:
            prior_layers: Callable returning the number of layers
            prior_hidden_size: Callable returning the hidden size
            prior_weight: Callable returning edge weights
            activation: Activation function
            outcome_noise_scale: Scale of the outcome noise
        """
        self.f_generator = BaseMLPGenerator(
            prior_layers, prior_hidden_size, prior_weight, activation,
            use_layer_noise=False
        )
        self.g_generator = BaseMLPGenerator(
            prior_layers, prior_hidden_size, prior_weight, activation,
            use_layer_noise=False
        )
        self.outcome_noise_scale = outcome_noise_scale

    def generate_outcomes(
        self, 
        X: np.ndarray, 
        A: np.ndarray, 
        U: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate outcomes for IV setting.
        
        Args:
            X: Observed covariates
            A: Treatment values
            U: Unobserved confounders
            
        Returns:
            Tuple of (Y, Y0, Y1) - observed and potential outcomes
        """
        n_samples = X.shape[0]

        # Setup f network (treatment effect)
        input_f = np.hstack([X, A.reshape(-1, 1)])
        n_features_f = input_f.shape[1]
        num_layers_f = max(3, self.f_generator.prior_layers())
        hidden_size_f = self.f_generator.prior_hidden_size()
        self.f_generator.network = self.f_generator._construct_network(
            num_layers_f, hidden_size_f, n_features_f, 1
        )
        self.f_generator._sample_network_parameters()

        # Setup g network (confounding effect)
        input_g = np.hstack([X, U])
        n_features_g = input_g.shape[1]
        num_layers_g = max(3, self.g_generator.prior_layers())
        hidden_size_g = self.g_generator.prior_hidden_size()
        self.g_generator.network = self.g_generator._construct_network(
            num_layers_g, hidden_size_g, n_features_g, 1
        )
        self.g_generator._sample_network_parameters()

        Y = np.zeros(n_samples)
        Y0 = np.zeros(n_samples)
        Y1 = np.zeros(n_samples)
        outcome_noise = np.random.normal(0, self.outcome_noise_scale, n_samples)

        for i in range(n_samples):
            g_val = self.g_generator._forward_propagate(input_g[i])
            f_val_factual = self.f_generator._forward_propagate(np.hstack([X[i], A[i]]))
            f_val_0 = self.f_generator._forward_propagate(np.hstack([X[i], 0]))
            f_val_1 = self.f_generator._forward_propagate(np.hstack([X[i], 1]))

            Y[i] = f_val_factual + g_val + outcome_noise[i]
            Y0[i] = f_val_0 + g_val + outcome_noise[i]
            Y1[i] = f_val_1 + g_val + outcome_noise[i]
            
        return Y, Y0, Y1


class IVDataGenerator(BaseDataGenerator):
    """
    Generator for Instrumental Variables datasets.
    
    Generates synthetic datasets with:
    - Observed covariates X
    - Unobserved confounders U
    - Instrument Z (binary or continuous)
    - Binary treatment A
    - Potential outcomes Y(0), Y(1)
    
    Example:
        >>> generator = IVDataGenerator(
        ...     num_samples=1024, 
        ...     num_features=10,
        ...     instrument_type='binary',
        ...     seed=42
        ... )
        >>> df = generator.generate()
        >>> print(df.columns)
        Index(['x0', ..., 'u0', ..., 'z', 'treatment', 'outcome', 'y0', 'y1', 'ite'], dtype='object')
    """
    
    def __init__(
        self,
        num_samples: int = 1024,
        num_features: int = 10,
        num_confounders_range: Tuple[int, int] = (2, 5),
        instrument_type: str = 'binary',
        seed: Optional[int] = None
    ):
        """
        Initialize the IV data generator.
        
        Args:
            num_samples: Number of samples per dataset
            num_features: Number of observed covariates
            num_confounders_range: Range for number of unobserved confounders
            instrument_type: 'binary' or 'continuous'
            seed: Random seed for reproducibility
        """
        super().__init__(num_samples, num_features, seed)
        self.num_confounders_range = num_confounders_range
        self.instrument_type = instrument_type
    
    def generate(self) -> pd.DataFrame:
        """
        Generate a single IV dataset.
        
        Returns:
            DataFrame with columns for X, U, Z, treatment, outcome, y0, y1, ite
        """
        # Generate covariates and confounders together
        num_confounders = np.random.randint(
            self.num_confounders_range[0], 
            self.num_confounders_range[1] + 1
        )
        num_total_features = self.num_features + num_confounders
        
        scm = DAGStructuredSCM()
        W = scm.generate_dataset(num_total_features, self.num_samples)
        
        U = W[:, :num_confounders]
        X = W[:, num_confounders:num_confounders + self.num_features]
        
        # Generate instrument
        instrument_gen = InstrumentGenerator()
        Z = instrument_gen.generate_instrument(X, self.instrument_type)
        
        # Generate treatment
        iv_treatment = IVTreatmentAssigner()
        A = iv_treatment.generate_treatments(X, Z, U)
        
        # Generate outcomes
        iv_outcome = IVOutcomeGenerator()
        Y, Y0, Y1 = iv_outcome.generate_outcomes(X, A, U)
        
        # Create DataFrame
        df_X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
        df_U = pd.DataFrame(U, columns=[f'u{i}' for i in range(U.shape[1])])
        df_Z = pd.DataFrame(Z.reshape(-1, 1), columns=['z'])
        df_A = pd.DataFrame(A, columns=['treatment'])
        df_Y = pd.DataFrame({
            'outcome': Y, 
            'y0': Y0, 
            'y1': Y1, 
            'ite': Y1 - Y0
        })
        
        return pd.concat([df_X, df_U, df_Z, df_A, df_Y], axis=1)
    
    def generate_multiple(
        self,
        num_datasets: int,
        output_dir: str,
        filename_prefix: str = "synthetic_iv_dataset"
    ) -> None:
        """
        Generate multiple IV datasets and save to files.
        
        Args:
            num_datasets: Number of datasets to generate
            output_dir: Directory to save the datasets
            filename_prefix: Prefix for the dataset filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {num_datasets} IV datasets...")
        
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

