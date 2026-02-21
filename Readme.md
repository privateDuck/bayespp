# BayesPP

**BayesPP** is a lightweight, header-only Bayesian Optimization library written in modern C++17. It is designed to find the global maximum of expensive-to-evaluate black-box functions.

It uses a **Matérn 5/2 kernel** for the Gaussian Process surrogate model and optimizes the **Expected Improvement (EI)** acquisition function to balance exploration and exploitation.

## ✨ Features
* **Header-only**: No build step required; just include the header and go.
* **Modern C++17**: Clean, safe, and intuitive interface.
* **Flexible Parameter Space**: Supports continuous (real, log-real), integer, categorical (option), and fixed parameters.
* **Intuitive API**: Just define your objective function as a functor, set your search bounds, and run the solver.
* **Powerful Backends**: Built on top of [Eigen](https://eigen.tuxfamily.org/) for fast linear algebra and [LBFGSpp](https://github.com/yixuan/LBFGSpp) for robust numerical optimization.

## 📦 Dependencies
* **C++17** compatible compiler
* **Eigen** (v3.3 or later)
* **LBFGSpp**

## 🚀 Installation
Since BayesPP is header-only, installation is trivial:
1. Ensure Eigen and LBFGSpp are in your include path.
2. Drop the `bayespp` folder into your project's include directory.
3. `#include "bayespp/BayesPP.hpp"` in your code.

## 📖 Quick Start

BayesPP requires you to define your objective function as a C++ functor (or class with an `operator()`). **Note: BayesPP always maximizes the objective.** If you want to minimize a function, simply return its negative value.

Here is an example of minimizing the famous [Branin function](https://www.sfu.ca/~ssurjano/branin.html) (by maximizing its negative):

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "bayespp/BayesPP.hpp"

// 1. Define the Objective Function to MAXIMIZE
class BraninFunction {
public:
    double operator()(const std::vector<double>& x) const {
        const double x1 = x[0];
        const double x2 = x[1];

        // Branin constants
        const double a = 1.0;
        const double b = 5.1 / (4.0 * M_PI * M_PI);
        const double c = 5.0 / M_PI;
        const double r = 6.0;
        const double s = 10.0;
        const double t = 1.0 / (8.0 * M_PI);

        // Calculate standard Branin (global minimum is ~0.397887)
        const double term1 = x2 - b * x1 * x1 + c * x1 - r;
        const double branin_val = a * term1 * term1 + s * (1.0 - t) * std::cos(x1) + s;

        // Return negative to find the minimum!
        return -branin_val; 
    }
private:
    static constexpr double M_PI = 3.141592653589793238;
};

int main() {
    BraninFunction objective_fn;

    // 2. Define the Search Space
    bayespp::ParameterSpace ps;
    ps.AddRealParameter(-5, 10); // x1 bounds
    ps.AddRealParameter(0, 15);  // x2 bounds

    // 3. Configure the Optimizer (Using default parameters)
    bayespp::BayesParameters opt_params;
    opt_params.max_iterations = 100;

    // 4. Initialize and Run
    bayespp::BayesOptimizer solver(ps, opt_params);
    std::vector<double> best_params;
    
    const double max_val = solver.Maximize(objective_fn, best_params);

    std::cout << "Best value found: " << -max_val << "\n";
    std::cout << "Best parameters: x1=" << best_params[0] 
              << ", x2=" << best_params[1] << "\n";

    return 0;
}
```

## 🛠️ Advanced Usage

### Defining the Parameter Space

bayespp::ParameterSpace allows you to define the exact nature of your search space. Parameters are passed to your objective function via std::vector<double> in the exact order they are added to the parameter space.

#### Continuous Parameters

* `AddRealParameter(min, max)`: A standard floating-point parameter bounded between min and max.

* `AddLogRealParameter(min, max)`: Ideal for parameters spanning multiple orders of magnitude (e.g., learning rates). The parameter is optimized in log-space (between log(min) and log(max)) and automatically transformed back before being passed to your objective function.

#### Discrete & Categorical Parameters

* `AddIntegerParameter(min, max)`: Functions identically to a real parameter but strictly constrained to integer values.

* `AddOptionParameter(num_options)`: Used for categorical choices (e.g., 0, 1, 2 representing "linear", "poly", "rbf"). The optimizer explores integers from 0 to num_options - 1. Your objective function should expect a double that can be safely cast to an int.

#### Fixed Parameters

If you want to temporarily lock a parameter without changing your objective function's signature or index mapping, you can use fixed parameters:

* `AddFixedRealParameter(value)`
* `AddFixedIntegerParameter(value)`
* `AddFixedOptionParameter(value)`

Fixed parameters are entirely ignored by the optimizer's search space but are faithfully injected into the std::vector<double> evaluated by your functor.

```cpp
bayespp::ParameterSpace ps;

// Real numbers in a linear space
ps.AddRealParameter(-5.0, 10.0);

// Real numbers in a log space (great for large ranges like learning rates)
ps.AddLogRealParameter(1e-5, 1e-1);

// Integer values
ps.AddIntegerParameter(2, 5);

// Categorical Options (e.g., 3 means indices 0, 1, 2)
// Your objective function should cast this double back to an int to handle logic.
ps.AddOptionParameter(3);

// Fixed parameters (useful for ablation studies without changing indices)
ps.AddFixedRealParameter(2.4);
ps.AddFixedIntegerParameter(4);
ps.AddFixedOptionParameter(1);
```

### Configuration & Tuning

The bayespp::BayesParameters struct allows you to fine-tune the behavior of the Bayesian optimizer. The default values are generally sufficient, but can be adjusted for complex topographies.

* `exploration_parameter` (Default: 0.01): Controls the exploration-exploitation trade-off in the Expected Improvement acquisition function.
Value can be interpreted as a probability.

* `max_evaluations` (Default: 100): The number of times the objective function will be evaluated.

* `n_initial_points` (Default: 5): The number of random initial points evaluated to bootstrap the Gaussian Process. Must be strictly greater than 2.

* `n_acq_samples` (Default: 1000): The number of random samples drawn from the acquisition function to find promising regions.

* `n_acq_restarts` (Default: 3): The number of the best candidates (peaks) from the max_ei_candidates pool that will be refined and optimized.

* `enable_kernel_multistart` (Default: true): If true, the library uses a multi-start optimizer to fit the GP kernel hyperparameters. This is computationally more expensive but yields a much more accurate surrogate model.

⚠️ Total number of function evaluations = `max_evaluations + n_initial_points`


```cpp
bayespp::BayesParameters opt_params;

// Trade-off: Higher means more exploration (searching new areas),
// Lower means more exploitation (refining known good areas). Default: 0.01
opt_params.exploration_parameter = 0.01;

// Total number of times your objective function will be called.
opt_params.max_evaluations = 100;

// Number of initial random evaluations to bootstrap the GP. Must be > 2.
opt_params.n_initial_points = 5;

// Number of random samples taken across the surrogate model to find the best Expected Improvement (EI).
opt_params.n_acq_samples = 1000;

// Number of the top EI candidates that will be optimized.
// Represents the number of potential peaks explored internally per iteration.
opt_params.n_acq_restarts = 3;

// True: Optimizes the kernel hyperparameters using multiple starts (expensive but more accurate).
// False: Uses a single start optimization.
opt_params.enable_kernel_multistart = true;
```