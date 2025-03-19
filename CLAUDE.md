# CLAUDE.md - Project Guidelines

## Environment & Commands
- Install dependencies: `poetry install`
- Run script: `poetry run python src/<file>.py`
- Lint code: `poetry run ruff check .`
- Type check: `poetry run mypy .`
- Test visualization: `poetry run python src/examples/<example_file>.py`

## Code Style Guidelines
- **Imports**: Standard library → third-party (jax, blackjax) → local imports
- **Types**: Use jaxtyping for arrays, include return types, leverage TypeVar for generics
- **Classes**: CamelCase, prefer frozen dataclasses for parameters, use Generic for type safety
- **Functions**: snake_case, define type aliases with CamelCase + "Fn" suffix
- **Patterns**: Functional programming, builder pattern, immutable data structures
- **Line Length**: 100 characters max
- **Error Handling**: Validate inputs with clear error messages, use assertions

## Project Architecture
- Probabilistic programming library with focus on Bayesian inference
- Uses JAX for numerical computing and automatic differentiation
- Organized by functionality (distributions, inference)
- Higher-order functions as primary abstraction
- Type-driven development with extensive type annotations