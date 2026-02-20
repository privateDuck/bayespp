#ifndef NLML_OPT_HPP
#define NLML_OPT_HPP
#pragma once

#include <iostream>
#include <LBFGSpp/LBFGSB.h>
#include "matern52.hpp"

namespace bayespp::detail {

    class GPObjective {
    public:
        GPObjective(const Eigen::MatrixXd& X_in, const Eigen::VectorXd& y_in, const double jitter_in = 1e-6)
            : X(X_in), y(y_in), jitter(jitter_in) {}

        double operator()(const Eigen::VectorXd& params, Eigen::VectorXd& grad) const {
            const Eigen::Index n = X.cols();

            // Exponentiate to enforce strict positivity
            const double sigma = std::exp(params[0]);
            const double l = std::exp(params[1]);
            const double sigma_sq = sigma * sigma;
            const double l_sq = l * l;

            const Matern52Kernel kern(sigma_sq, l_sq);

            Eigen::MatrixXd SqDist(n, n);
            Eigen::MatrixXd K = kern.Covariance(X, X, SqDist);

            // Add jitter to the diagonal for numerical stability during Cholesky
            K.diagonal().array() += jitter;

            // Cholesky decomp (K = L * L^T)
            const Eigen::LLT<Eigen::MatrixXd> llt(K);
            if (llt.info() == Eigen::NumericalIssue) {
                // If decomposition fails, return a high penalty to push the optimizer away
                return std::numeric_limits<double>::infinity();
            }

            // Compute alpha = K^{-1} y
            Eigen::VectorXd alpha = llt.solve(y);

            // Compute the Negative Log Marginal Likelihood (NLML)
            // Log Det 2 * sum(log(diag(L)))
            double log_det = 0.0;
            for (int i = 0; i < n; ++i) {
                log_det += 2.0 * std::log(llt.matrixL()(i, i));
            }

            const double data_fit = 0.5 * y.dot(alpha);
            const double complexity_penalty = 0.5 * log_det;
            const double constant = 0.5 * static_cast<double>(n) * std::log(6.283185307179586476925286766559); // 0.5*n*log(2*pi)

            const double nlml = data_fit + complexity_penalty + constant;

            // Compute Gradients
            // We need K^{-1} to compute W
            const Eigen::MatrixXd K_inv = llt.solve(Eigen::MatrixXd::Identity(n, n));

            // W = alpha * alpha^T - K^{-1}
            Eigen::MatrixXd W = alpha * alpha.transpose() - K_inv;

            // Gradient w.r.t log(sigma)
            grad[0] = static_cast<double>(n) - y.dot(alpha) + jitter * W.trace();

            // Gradient w.r.t log(l)
            double grad_l_sum = 0.0;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    // Multiply by -0.5 because NLML is Negative Log Marginal Likelihood
                    grad_l_sum += -0.5 * W(i, j) * kern.LGradientWeight(SqDist(i, j));
                }
            }
            grad[1] = grad_l_sum;

            return nlml;
        }

    private:
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        double jitter;
    };

    [[nodiscard]] inline Matern52Kernel ComputeOptimalKernel(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const double jitter = 1e-6) {
        LBFGSpp::LBFGSBParam<double> param_opt;
        param_opt.epsilon = 1e-4;
        param_opt.max_iterations = 100;

        LBFGSpp::LBFGSBSolver<double> solver(param_opt);
        GPObjective obj(X, y, jitter);
        Eigen::VectorXd param(2);
        param << 0.0,-0.693;

        // sigma can be exp(-3), l can be log(0.01) = -4.6
        Eigen::VectorXd lb(2);
        lb << -3.0, -4.6;

        // sigma capped at exp(2), l capped at log(10) approx 2.3
        Eigen::VectorXd ub(2);
        ub << 2.0, 2.3;

        try {
            double nlml;
            solver.minimize(obj, param, nlml, lb, ub);
        }
        catch (std::exception& e) {
            std::cerr << "NLML OPT: " << e.what() << std::endl;
        }

        return {std::exp(2.0 * param[0]), std::exp(2.0 * param[1])};
    }

    [[nodiscard]] inline Matern52Kernel ComputeOptimalKernelMultiStart(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const double jitter = 1e-4) {
    LBFGSpp::LBFGSBParam<double> param_opt;
    param_opt.epsilon = 1e-4;
    param_opt.max_iterations = 100;

    LBFGSpp::LBFGSBSolver<double> solver(param_opt);
    GPObjective obj(X, y, jitter);

    Eigen::VectorXd lb(2);
    lb << -3.0, -4.6;
    Eigen::VectorXd ub(2);
    ub << 2.0, 2.3;

    // grid of distinct initial guesses: {log_sigma, log_l}
    std::vector<Eigen::VectorXd> initial_guesses = {
        (Eigen::VectorXd(2) << 0.0, -0.693).finished(),   // Default (sigma=1, l=0.5)
        (Eigen::VectorXd(2) << -1.0, 0.0).finished(),     // Low variance, high length scale
        (Eigen::VectorXd(2) << 1.0, -2.0).finished(),     // High variance, low length scale
        (Eigen::VectorXd(2) << 0.0, 1.0).finished(),      // Standard variance, very high length scale
        (Eigen::VectorXd(2) << -2.0, -3.0).finished()     // Very low variance, very low length scale
    };

    double best_nlml = std::numeric_limits<double>::infinity();
    Eigen::VectorXd best_param = initial_guesses[0];

    for (const auto& guess : initial_guesses) {
        Eigen::VectorXd current_param = guess;
        try {
            double nlml;
            solver.minimize(obj, current_param, nlml, lb, ub);

            if (nlml < best_nlml) {
                best_nlml = nlml;
                best_param = current_param;
            }
        } catch (...) {
            // Line search failed for this specific start point due to a bad local valley
        }
    }

    // Fallback safely to the default guess
    if (std::isinf(best_nlml)) {
        best_param = initial_guesses[0];
    }

    return {std::exp(2.0 * best_param[0]), std::exp(2.0 * best_param[1])};
}

};
#endif // NLML_OPT_HPP