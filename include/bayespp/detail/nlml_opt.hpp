// Copyright (c) 2026 Jayawardane
// SPDX-License-Identifier: MIT
//
// This file is part of bayespp.
// See the LICENSE file in the project root for full license information.

#ifndef BAYESPP_NLML_OPT_HPP
#define BAYESPP_NLML_OPT_HPP
#pragma once

#include <iostream>
#include <LBFGSpp/LBFGSB.h>
#include "matern52.hpp"

namespace bayespp::detail {

    class GPObjective {
    public:
        GPObjective(
            const Eigen::MatrixXd& X_in,
            const Eigen::VectorXd& y_in,
            const double jitter_in = 1e-6
            )
        : X(X_in), y(y_in), jitter(jitter_in), n(X_in.cols()),
      SqDist(n, n), K(n, n), K_inv(n, n), W(n, n), alpha(n)
        {}

        double operator()(const Eigen::VectorXd& params, Eigen::VectorXd& grad) const {
            const double sigma_sq = std::exp(2.0 * params[0]);
            const double l_sq = std::exp(2.0 * params[1]);
            const Matern52Kernel kern(sigma_sq, l_sq);

            for (Eigen::Index i = 0; i < n; i++) {
                for (Eigen::Index j = 0; j < n; j++) {
                    K(i, j) = kern.Kernel(X.col(i), X.col(j), SqDist(i, j));
                }
            }

            // For numerical stability
            K.diagonal().array() += jitter;

            // Cholesky decomp (K = L * L^T)
            llt.compute(K);
            if (llt.info() == Eigen::NumericalIssue) {
                return std::numeric_limits<double>::infinity();
            }

            // Compute alpha = K^{-1} y
            alpha = y;
            llt.solveInPlace(alpha);

            // We need K^{-1} to compute W
            K_inv.setIdentity();
            llt.solveInPlace(K_inv);

            // W = alpha * alpha^T - K^{-1}
            W.noalias() = alpha * alpha.transpose() - K_inv;

            double log_det = 0.0;
            for (int i = 0; i < n; ++i) {
                log_det += 2.0 * std::log(llt.matrixL()(i, i));
            }

            // Compute the Negative Log Marginal Likelihood (NLML)
            const double nlml = 0.5 * y.dot(alpha) + 0.5 * log_det + 0.5 * static_cast<double>(n) * 1.8378770664093453;

            // Compute Gradients
            // Gradient w.r.t log(sigma)
            grad[0] = static_cast<double>(n) - y.dot(alpha) + jitter * W.trace();

            // Gradient w.r.t log(l)
            double grad_l_sum = 0.0;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
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
        Eigen::Index n;

        mutable Eigen::MatrixXd SqDist;
        mutable Eigen::MatrixXd K;
        mutable Eigen::MatrixXd K_inv;
        mutable Eigen::MatrixXd W;
        mutable Eigen::VectorXd alpha;
        mutable Eigen::LLT<Eigen::MatrixXd> llt;
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