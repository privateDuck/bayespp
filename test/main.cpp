

#include <iostream>
#include "bayespp/bayes.hpp"

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

struct Simple {
    double operator()(const std::vector<double>& param) const {
        return 2.0-param[0]*param[0];
    }
};

int main() {
    const double a = 1.0;
    const double b = 100.0;

    // The global maximum at (a,a*a)
    // We set the bounds to {(-2*a,2*a),(-2*a*a,2*a*a)}
    Rosenbrock fn(a,b);
    bayespp::ParameterSpace ps;
    ps.add_parameter(bayespp::ParameterSpace::Parameter::MakeReal(-2*a, 2*a));
    ps.add_parameter(bayespp::ParameterSpace::Parameter::MakeReal(-2*a*a, 2*a*a));
    bayespp::BayesParameters opt_params; //defaults
    opt_params.max_iterations = 50;
    opt_params.exploration_parameter = 0.01;
    std::cout << "Optimizing..." << std::endl;
    bayespp::BayesOptimizer solver(ps, opt_params);

    for (int i = 0; i < 10; ++i) {
        std::vector<double> params;
        double fxy_at_maximum = solver.Maximize(fn, params);

        std::cout << "f(x,y): " << fxy_at_maximum << '\n';
        std::cout << "x, y: "<< params[0] << ", " << params[1] << "\n\n";
    }
}