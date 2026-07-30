# Bayesian Optimization Demo

This repository demonstrates how to use a Gaussian Process Surrogate model to perform Bayesian optimization.

## Use Case: Semiconductor Process Optimization

In this example case, the application attempts to optimize process yield by searching for the ideal **Chamber Temperature (°C)**. The simulation includes a nonlinear base yield, a parabolic penalty moving away from the optimal point (~75°C), and some sinusoidal noise, reflecting real-world variability in manufacturing.

The application implements the **Upper Confidence Bound (UCB)** acquisition function to balance exploration (checking regions with high uncertainty) and exploitation (checking regions with high predicted yield).

## Demo

![Semiconductor Optimization Demo](demo.png)

## Getting Started

1. Initialize the modules:
   ```bash
   go mod init bayesopt
   go get gonum.org/v1/gonum/...
   ```
2. Build and run the app:
   ```bash
   go build main.go
   ./main
   ```
3. Open `http://localhost:8080` in your web browser.
