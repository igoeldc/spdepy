# spdepy Development Roadmap

## Stage 0 — Infrastructure
- [x] Set up Python package configuration and CI/CD
- [x] Configure testing and linting
- [x] Configure PyPI publishing

## Stage 1 — Mathematical Core
- [x] Define the broad mathematical scope
- [ ] Specify initial mathematical assumptions
- [ ] Design core mathematical interfaces
- [ ] Implement domains and boundary conditions
- [ ] Implement eigenbases and differential operators
- [ ] Implement noise and covariance structures
- [ ] Implement spectral Galerkin projection and reconstruction
- [ ] Implement finite-dimensional stochastic model representations

## Stage 2 — Minimum Viable Simulator
- [ ] Implement exact linear stochastic time integration
- [ ] Implement the stochastic heat equation
- [ ] Implement simulation and result interfaces
- [ ] Implement basic visualization
- [ ] Validate against analytical benchmarks
- [ ] Publish the first functional release

## Stage 3 — General Linear SPDEs
- [ ] Support arbitrary intervals and additional boundary conditions
- [ ] Support fractional operators and operator arithmetic
- [ ] Support additional noise and covariance structures
- [ ] Implement general linear transition calculations
- [ ] Implement additional stochastic time integrators
- [ ] Implement Monte Carlo simulation
- [ ] Implement numerical convergence experiments

## Stage 4 — Statistical Inference and Filtering
- [ ] Implement observation models
- [ ] Implement linear Gaussian state-space representations
- [ ] Implement Kalman filtering and smoothing
- [ ] Implement maximum-likelihood estimation
- [ ] Implement additional parameter estimators
- [ ] Implement parameter recovery and filtering experiments
- [ ] Validate statistical methods against analytical benchmarks

## Stage 5 — Nonlinear SPDEs
- [ ] Implement Nemytskii operators
- [ ] Support nonlinear drift
- [ ] Support multiplicative noise
- [ ] Implement suitable nonlinear stochastic integrators
- [ ] Implement nonlinear benchmark SPDEs

## Stage 6 — Advanced Numerical Methods
- [ ] Implement finite difference methods
- [ ] Support higher-dimensional domains
- [ ] Support more general differential operators
- [ ] Investigate finite element methods
- [ ] Implement sparse numerical representations

## Stage 7 — Advanced Inference and Filtering
- [ ] Implement SPDE-specific statistical estimators
- [ ] Implement nonlinear filtering methods
- [ ] Implement ensemble and particle filtering
- [ ] Implement Bayesian inference
- [ ] Support joint state and parameter estimation
- [ ] Implement statistical convergence and asymptotic experiments

## Stage 8 — Long-Term Extensions
- [ ] Support advanced SPDE systems
- [ ] Support additional stochastic forcing structures
- [ ] Implement advanced visualization
- [ ] Optimize computational performance
- [ ] Support custom numerical and statistical methods
- [ ] Reproduce published research benchmarks

## Ongoing
- [ ] Maintain mathematical documentation
- [ ] Maintain API documentation and tutorials
- [ ] Add tests alongside new functionality
- [ ] Maintain numerical and statistical validation
- [ ] Maintain backward compatibility where practical