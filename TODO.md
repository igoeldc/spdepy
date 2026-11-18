# TODO

## Infrastructure
- [x] Set up Python package configuration, testing, and CI/CD

## Mathematical Foundations
- [x] Define the broad mathematical scope of the library
- [ ] Specify the initial SPDE, assumptions, and solution concept
- [ ] Choose spatial discretization and stochastic time-integration methods
- [ ] Define supported noise types and covariance structures
- [ ] Define observation models for inference and filtering

## Library Design
- [ ] Design interfaces for domains, operators, and boundary conditions
- [ ] Design interfaces for SPDE equations and stochastic forcing
- [ ] Design interfaces for solvers and simulation results
- [ ] Design interfaces for observations, parameter estimation, and filtering

## Simulation
- [ ] Implement the one-dimensional Dirichlet Laplacian and its eigenbasis
- [ ] Implement spectral Galerkin discretization
- [ ] Implement additive Gaussian noise with configurable covariance
- [ ] Implement the linear stochastic heat equation
- [ ] Implement exact modal time stepping
- [ ] Add solution visualization

## Statistical Inference
- [ ] Implement observation models for discretely sampled SPDE solutions
- [ ] Implement likelihood-based parameter estimation for linear Gaussian models
- [ ] Support estimation of drift and noise parameters
- [ ] Add parameter recovery experiments using simulated observations

## Filtering
- [ ] Implement linear Gaussian state-space representations of discretized SPDEs
- [ ] Implement exact discrete-time transition and covariance calculations
- [ ] Implement Kalman filtering
- [ ] Add filtering experiments with noisy and partial observations
- [ ] Evaluate state estimation accuracy and uncertainty calibration

## Validation
- [ ] Add reproducible numerical tests with seeded randomness
- [ ] Validate simulated moments against analytical results
- [ ] Implement spatial and temporal convergence experiments
- [ ] Validate parameter estimation and filtering against known benchmarks
- [ ] Document mathematical assumptions, numerical methods, and limitations

## Future Extensions
- [ ] Support fractional powers of the Laplacian
- [ ] Support additional domains and boundary conditions
- [ ] Support nonlinear drift terms (Nemytskii operators)
- [ ] Support multiplicative noise
- [ ] Investigate nonlinear and non-Gaussian filtering methods
- [ ] Investigate inference for partially observed nonlinear SPDEs