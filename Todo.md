Probabl Framework Development Blueprint

  Phase 1: Context-Aware Data Function Implementation

  Step 1: Create Data Context Types and Enums

  ## LLM Prompt: Data Context Implementation

  Create the foundation for context-aware data functions by implementing:

  1. A `DataContext` enum with types:
     - `PRIOR_PREDICTIVE`
     - `INFERENCE`
     - `POSTERIOR_PREDICTIVE`

  2. A `DataContextManager` class that:
     - Tracks the current context
     - Provides a context stack for nested contexts
     - Exposes context getter/setter methods
     - Implements context manager protocol for use with `with` statements

  3. Create a module-level default context manager instance

  The code should follow the project's functional style with immutable data structures and type annotations.

  Create this in a new file at: `src/inference/context.py`

  Step 2: Create Enhanced Data Function Type

  ## LLM Prompt: Context-Aware Data Function Type

  Enhance the data function abstractions in `src/distributions/continous.py` to support context awareness:

  1. Extend the existing `Data` type alias to create a new `ContextAwareData` type that:
     - Has the same function signature as the existing `Data` type
     - Can optionally accept a `DataContext` parameter
     - Returns appropriate data based on the current context

  2. Create a `make_context_aware_data` function that:
     - Takes separate data functions for different contexts
     - Returns a unified data function that switches behavior based on the current context
     - Falls back to sensible defaults if a specific context handler is not provided

  3. Update the relevant type annotations and add appropriate documentation

  The implementation should maintain backward compatibility with existing code.

Step 3: Implement Prior Simulation Utilities

  ## LLM Prompt: Prior Simulation Utilities

  Create utilities for generating prior simulations in `src/inference/simulation.py`:

  1. Implement a `simulate_from_prior` function that:
     - Takes a prior distribution (ParametricDensityFn)
     - Takes a JAX random key for reproducibility
     - Takes optional shape parameters
     - Uses rejection sampling or other appropriate method to generate samples
     - Returns an array of samples from the prior

  2. Implement a `data_from_prior` factory function that:
     - Takes a prior and turns it into a data provider function
     - Can be used in the context-aware data system
     - Caches generated data for efficiency when appropriate

  3. Add appropriate type annotations and tests

  This implementation should align with the functional programming style of the existing codebase.

  Phase 2: Enhanced Model Building

  Step 4: Create Unified Model Builder

  ## LLM Prompt: Enhanced Model Builder Interface

  Enhance the `ProbabilisticModelBuilder` in `src/distributions/continous.py` to support the context-aware data framework:

  1. Update the builder to accept context-aware data functions
  2. Add methods for specifying different data providers for different contexts:
     - `with_observed_data(data_fn)` - for inference context
     - `with_prior_simulator(prior_fn, rng_key)` - for prior predictive context
     - `with_posterior_simulator(posterior_fn, samples)` - for posterior predictive context

  3. Modify the `build()` method to appropriately construct a unified context-aware data function

  4. Add helper methods for common model construction patterns

  5. Ensure all methods maintain the fluent interface pattern and return self for method chaining

  The enhanced builder should be backward compatible with existing code while enabling the new context-aware functionality.

  Step 5: Parameter Structure Handling

  ## LLM Prompt: Parameter Flattening and Unflattening Utilities

  Create utilities for handling structured parameters in `src/inference/parameters.py`:

  1. Implement parameter tree flattening utilities:
     - `flatten_params(params)` - converts nested parameter structures to flat arrays
     - `unflatten_params(flat_params, structure)` - restores original structure

  2. Create a `ParameterStructure` class that:
     - Captures the structure of parameters
     - Provides methods to convert between flat arrays and structured parameters
     - Can be serialized/deserialized for storage

3. Add transformation utilities for constrained parameters:
     - `transform_params(params, constraints)` - applies transformations to enforce constraints
     - `untransform_params(transformed, constraints)` - reverses transformations

  4. Ensure compatibility with JAX's transformations

  Implement with appropriate type annotations and following the project's functional style.

  Phase 3: Workflow Integration

  Step 6: Implement Model Checking Utilities

  ## LLM Prompt: Prior and Posterior Predictive Checking Utilities

  Create model checking utilities in `src/inference/checking.py`:

  1. Implement prior predictive checking utilities:
     - `check_prior_predictive(model, n_samples)` - generates samples from prior predictive
     - `summarize_prior_predictive(samples, observed_data)` - compares simulation to observed data

  2. Implement posterior predictive checking utilities:
     - `check_posterior_predictive(model, posterior_samples)` - generates predictive samples
     - `summarize_posterior_predictive(samples, observed_data)` - compares to observed data

  3. Add visualization helpers for common diagnostics:
     - Histograms comparing simulated vs observed data
     - Q-Q plots for distribution comparisons
     - Summary statistics comparisons

  4. Integrate with the context system to automatically use the right data context

  Ensure all functions have appropriate type annotations and follow the functional programming style of the project.

  Step 7: Bayesian Workflow Manager

  ## LLM Prompt: Unified Bayesian Workflow Manager

  Create a workflow manager class in `src/inference/workflow.py` that orchestrates the entire Bayesian workflow:

  1. Implement a `BayesianWorkflow` class that:
     - Takes a model built with the enhanced builder
     - Manages transitions between workflow phases
     - Provides helper methods for each phase of the workflow

  2. Implement methods for each workflow phase:
     - `prior_check()` - runs prior predictive checks
     - `run_inference()` - performs posterior sampling
     - `posterior_check()` - runs posterior predictive checks

  3. Add state tracking to maintain results from each phase

  4. Add helper methods for visualizing results from each phase

  5. Implement automatic context management for each phase

  Make the implementation highly modular, with appropriate type annotations and following functional patterns where possible.
  
Phase 4: Integration and Examples

  Step 8: Refactor Existing 8 Schools Example

  ## LLM Prompt: Refactor 8 Schools Example with Context-Aware Framework

  Refactor the existing 8 schools example to use the new context-aware framework in `src/examples/eight_schools_context_aware.py`:

  1. Use the enhanced builder to create the model
  2. Explicitly define data functions for different contexts:
     - Observed data for inference
     - Prior simulator for prior predictive checks
     - Posterior simulator for posterior predictive checks

  3. Use the workflow manager to orchestrate the full analysis:
     - Prior predictive checking
     - MCMC inference
     - Posterior predictive checking

  4. Add visualizations to compare results across phases

  5. Add detailed comments explaining how the context-aware framework enhances the analysis

  The refactored example should demonstrate all major features of the new framework while maintaining the core statistical approach of the original.

  Step 9: Create Linear Regression Example

  ## LLM Prompt: Linear Regression Example with Context-Aware Framework

  Create a comprehensive linear regression example in `src/examples/linear_regression.py` that demonstrates the full power of the context-aware framework:

  1. Implement a Bayesian linear regression model with:
     - Priors on coefficients and error variance
     - Normal likelihood for observations
     - Support for simulating from the prior model

  2. Generate synthetic data with known parameters for validation

  3. Implement the full Bayesian workflow:
     - Prior predictive checks to validate model assumptions
     - MCMC inference to estimate parameters
     - Posterior predictive checks to validate fit

  4. Add visualizations for each step:
     - Prior predictive distributions
     - Trace plots for MCMC diagnostics
     - Posterior parameter distributions
     - Posterior predictive distributions vs observed data

  5. Add detailed documentation as a tutorial for users

  Include appropriate type annotations and follow the project's functional style.

Step 10: Create Hierarchical Model Example

  ## LLM Prompt: Hierarchical Model Example with Context-Aware Framework

  Create a hierarchical modeling example in `src/examples/hierarchical_model.py` that demonstrates how the framework handles complex model structures:

  1. Implement a multi-level hierarchical model with:
     - Group-level parameters with hyperpriors
     - Individual-level parameters with group-level priors
     - Observations with likelihood based on individual parameters

  2. Generate synthetic hierarchical data for demonstration

  3. Implement the full Bayesian workflow:
     - Prior simulation to validate hierarchical structure
     - MCMC inference with appropriate diagnostics for hierarchical models
     - Posterior predictive checks at multiple levels

  4. Add visualizations that highlight:
     - Partial pooling behavior
     - Shrinkage effects
     - Group-level vs individual-level parameters

  5. Include comprehensive documentation explaining the benefits of the context-aware approach for hierarchical models

  Ensure code is well-typed and follows the project's functional programming style.


Phase 5: Advanced Features

  Step 11: Auto-tuning MCMC Parameters

  ## LLM Prompt: Auto-tuning MCMC Parameters

  Enhance the MCMC sampling capabilities in `src/inference/samplers.py`:

  1. Expand `nuts_with_warmup` to include:
     - Dynamic adaptation period based on model complexity
     - Progress reporting and diagnostics during warmup
     - Early stopping when adaptation stabilizes

  2. Add parameters for controlling adaptation:
     - Target acceptance rate adjustment
     - Mass matrix estimation method selection
     - Step size jitter controls

  3. Implement diagnostics tools:
     - Effective sample size monitoring
     - R-hat convergence statistics
     - Energy transition statistics
     - Tree depth warnings

  4. Add automatic diagnostics reporting to highlight potential issues

  Ensure implementation is compatible with existing code and includes appropriate type annotations.

  Step 12: Implement Model Comparison Tools

  ## LLM Prompt: Bayesian Model Comparison Tools

  Create tools for Bayesian model comparison in `src/inference/comparison.py`:

  1. Implement information criteria calculations:
     - WAIC (Widely Applicable Information Criterion)
     - LOO-CV (Leave-One-Out Cross-Validation)
     - DIC (Deviance Information Criterion)

  2. Add Bayes factor approximation methods:
     - Bridge sampling
     - Thermodynamic integration

  3. Create utilities for predictive performance metrics:
     - PSIS-LOO predictive score
     - Expected log predictive density

  4. Add visualization tools for model comparison:
     - WAIC comparison plots
     - Predictive performance comparison
     - Parameter stability comparison

  Implement with appropriate type annotations and in a style consistent with the existing codebase.


Step 13: Data Function Transformation Utilities

  ## LLM Prompt: Data Function Transformation Utilities

  Create utilities for transforming data functions in `src/inference/transformations.py`:

  1. Implement data transformations that preserve context awareness:
     - `map_data(data_fn, transform_fn)` - applies a transform to the output of a data function
     - `filter_data(data_fn, predicate_fn)` - filters data based on a predicate
     - `batch_data(data_fn, batch_size)` - creates mini-batches for stochastic gradient methods

  2. Add utilities for common statistical transformations:
     - Standardization (z-scoring)
     - Normalization (min-max scaling)
     - Log and other nonlinear transformations

  3. Create a `DataTransformer` class that:
     - Tracks transformations applied to data
     - Can apply and unapply transformations
     - Preserves context awareness

  4. Ensure all transformations work with JAX's automatic differentiation

  Maintain type safety and follow the project's functional programming style.

  Phase 6: Documentation and Testing

  Step 14: Documentation Generator

  ## LLM Prompt: Documentation Generation System

  Create a comprehensive documentation system for the library:

  1. Implement a documentation generator in `scripts/generate_docs.py` that:
     - Extracts type signatures and docstrings
     - Generates markdown documentation
     - Creates interactive examples in Jupyter notebook format

  2. Create documentation templates for:
     - Core concepts
     - API reference
     - Tutorials with examples
     - Advanced topics

  3. Add docstrings to all public functions and classes

  4. Create a documentation website structure using a static site generator

  The generator should be designed to automatically update documentation as the codebase evolves.
  
Step 15: Test Suite Enhancement

  ## LLM Prompt: Comprehensive Test Suite Implementation

  Create a comprehensive test suite for the library:

  1. Implement unit tests for core functionality:
     - Context management system
     - Data transformation utilities
     - Parameter handling
     - Model building and composition

  2. Add integration tests for workflow components:
     - End-to-end Bayesian workflow testing
     - MCMC sampling with known distributions
     - Model checking with simulated data

  3. Create property-based tests to verify:
     - Type safety
     - Function composition properties
     - Invariants under transformations

  4. Implement benchmark tests to monitor performance

  5. Add CI/CD integration for automated testing

  Tests should be organized to match the module structure and provide good coverage of all functionality.
  
Review and Refinement Steps

  This blueprint provides a detailed, step-by-step approach to implementing the Probabl framework as outlined in the idea.md document. The steps are broken down into manageable
  chunks that build upon each other while maintaining a coherent direction toward the final goal.

  Each step is detailed enough to guide implementation but leaves room for creative solutions and adaptations as the project evolves. The prompts are designed to direct an LLM
  to produce high-quality, type-safe code that follows the functional programming paradigm established in the existing codebase.

  For implementation, I recommend following these phases sequentially, as each builds upon the previous. However, within each phase, some steps could be tackled in parallel if
  multiple developers are working on the project.

  The most critical components are the context-aware data functions (Phase 1) and the enhanced model builder (Phase 2), as these form the foundation of the framework's
  innovative approach. Once these are in place, the workflow integration (Phase 3) and examples (Phase 4) will help validate the design and demonstrate its benefits.
