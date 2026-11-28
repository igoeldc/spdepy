# spdepy Development Roadmap

## Stage 0 — Infrastructure
- [x] Set up Python package configuration and CI/CD
- [x] Configure testing and linting
- [x] Configure PyPI publishing

## Stage 1 — Mathematical Core
- [x] Define the broad mathematical scope
- [x] Specify initial mathematical assumptions
- [x] Design core mathematical interfaces
- [x] Implement domains and boundary conditions
- [x] Implement eigenbases and differential operators
- [x] Implement noise and covariance structures
- [x] Implement spectral Galerkin projection and reconstruction
- [x] Implement finite-dimensional stochastic model representations
- [x] Implement centralized input validation

## Stage 2 — Minimum Viable Simulator
- [x] Implement exact linear stochastic time integration
- [x] Implement the stochastic heat equation
- [x] Implement single-path simulation
- [x] Support explicit, fixed-step, and fixed-count temporal grids
- [x] Support spectral and callable initial conditions
- [ ] Implement simulation result interfaces
- [ ] Implement spatiotemporal solution reconstruction
- [ ] Implement basic visualization
- [ ] Validate against analytical benchmarks
- [ ] Finalize the initial public API
- [ ] Create introductory examples and tutorials
- [ ] Publish the first functional release (v0.1.0)

## Stage 3 — General Linear SPDEs
- [x] Support arbitrary bounded intervals
- [ ] Support additional boundary conditions
- [ ] Support fractional operators and operator arithmetic
- [ ] Support additional noise and covariance structures
- [ ] Support non-diagonal spectral covariance structures
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