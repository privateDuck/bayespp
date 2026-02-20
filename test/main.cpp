
#include <chrono>
#include <iostream>
#include "bayespp/BayesPP.hpp"

struct Rosenbrock {
    Rosenbrock(double a, double b) : a_(a), b_(b) {}

    double operator()(const std::vector<double>& param) const {
        const double x = param[0];
        const double y = param[1];
        return -((a_ - x)*(a_ - x) + b_ * (y - x*x)*(y - x*x));
    }

    [[nodiscard]] std::vector<double> minimum() const {
        return {a_, a_*a_};
    }
private:
    double a_, b_;
};

class BraninFunction {
public:
    double operator()(const std::vector<double>& x) const {
        // Map normalized space to Branin's true domain:
        // x1 is in [-5, 10]
        // x2 is in [0, 15]
        const double x1 = x[0];
        const double x2 = x[1];

        // Branin constants
        const double a = 1.0;
        const double b = 5.1 / (4.0 * M_PI * M_PI);
        const double c = 5.0 / M_PI;
        const double r = 6.0;
        const double s = 10.0;
        const double t = 1.0 / (8.0 * M_PI);

        // Calculate standard Branin (minimum is ~0.397887)
        const double term1 = x2 - b * x1 * x1 + c * x1 - r;
        const double branin_val = a * term1 * term1 + s * (1.0 - t) * std::cos(x1) + s;

        return -branin_val;
    }

private:
    static constexpr double M_PI = 3.141592653589793238;
};

void test_branin() {
    BraninFunction fn;
    bayespp::ParameterSpace ps;
    // Branin's domain:
    // x1 is in [-5, 10]
    // x2 is in [0, 15]
    ps.AddRealParameter(-5, 10);
    ps.AddRealParameter(0, 15);
    bayespp::BayesParameters opt_params; //defaults
    opt_params.max_iterations = 50;
    opt_params.exploration_parameter = 0.01;
    opt_params.use_multistart_kernel_optimizer = true;

    std::cout << "Optimizing Branin..." << std::endl;
    bayespp::BayesOptimizer solver(ps, opt_params);

    for (int i = 0; i < 10; ++i) {
        std::vector<double> params;
        const double fxy_at_maximum = solver.Maximize(fn, params);

        std::cout << "f(x,y): " << fxy_at_maximum << '\n';
        std::cout << "x, y: "<< params[0] << ", " << params[1] << "\n\n";
    }
}

void test_rosenbrock() {
    constexpr double a = 1.0;
    constexpr double b = 100.0;

    // The global maximum at (a,a*a)
    // We set the bounds to {(-2*a,2*a),(-2*a*a,2*a*a)}
    Rosenbrock fn(a,b);
    bayespp::ParameterSpace ps;
    ps.AddRealParameter(-2*a, 2*a);
    ps.AddRealParameter(-2*a*a, 2*a*a);
    bayespp::BayesParameters opt_params; //defaults
    opt_params.max_iterations = 50;
    opt_params.exploration_parameter = 0.01;
    std::cout << "Optimizing Rosenbrock..." << std::endl;
    bayespp::BayesOptimizer solver(ps, opt_params);

    for (int i = 0; i < 10; ++i) {
        std::vector<double> params;
        const double fxy_at_maximum = solver.Maximize(fn, params);

        std::cout << "f(x,y): " << fxy_at_maximum << '\n';
        std::cout << "x, y: "<< params[0] << ", " << params[1] << "\n\n";
    }
}

int main() {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    test_branin();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    const double time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * 1e-6;
    std::cout << "Time took (us): " << time << std::endl;
}