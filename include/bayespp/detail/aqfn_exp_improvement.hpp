// Copyright (c) 2026 Jayawardane
// SPDX-License-Identifier: MIT
//
// This file is part of bayespp.
// See the LICENSE file in the project root for full license information.

#ifndef BAYESPP_AQFN_EXP_IMPROV_HPP
#define BAYESPP_AQFN_EXP_IMPROV_HPP
#pragma once

#include <LBFGSpp/LBFGSB.h>
#include "matern52.hpp"

namespace bayespp::detail {

    struct EI_Candidate {
        Eigen::VectorXd candidate;
        double y;

        friend bool operator<(const EI_Candidate& lhs, const EI_Candidate& rhs)
        {
            return lhs.y < rhs.y;
        }

        friend bool operator>(const EI_Candidate& lhs, const EI_Candidate& rhs)
        {
            return lhs.y > rhs.y;
        }
    };

    class ExpectedImprovement {
    public:
        explicit ExpectedImprovement(
            const Matern52Kernel& kernel,
            const Eigen::MatrixXd& X,
            const Eigen::MatrixXd& Y,
            const double fx_best,
            const double exploration = 0.01
            )
        : kernel(kernel), X_(X), Y_(Y), fx_best(fx_best), exploration(exploration),
        k_star(X.cols()), imm(X.cols()), grad_k_star(X.rows(), X.cols()),
        grad_mu(X.rows()), grad_sigma(X.rows())
        {
            Eigen::MatrixXd K(X.cols(), X.cols());
            for (Eigen::Index i = 0; i < X_.cols(); i++) {
                for (Eigen::Index j = 0; j < X_.cols(); j++) {
                    K(i, j) = kernel.Kernel(X_.col(i), X_.col(j));
                }
            }
            K.diagonal().array() += 1e-6;
            llt.compute(K);
            alpha = llt.solve(Y_);
        }

        // EI(X)
        double operator()(const Eigen::VectorXd& x) const {
            for (Eigen::Index i = 0; i < X_.cols(); i++) {
                k_star(i) = kernel.Kernel(X_.col(i), x);
            }
            const double kb = kernel.Kernel(x, x);
            const double muX = k_star.dot(alpha);

            imm = k_star;
            llt.solveInPlace(imm);

            const double variance = std::max(0.0, kb - k_star.dot(imm));
            const double sigmaX = std::sqrt(variance);

            if (sigmaX < 1e-9) {
                return 0.0;
            }

            const double Z = (muX - fx_best - exploration) / sigmaX;
            const double cdf_Z = norm_cdf(Z);
            const double pdf_Z = norm_pdf(Z);
            return (muX - fx_best - exploration) * cdf_Z + sigmaX * pdf_Z;
        }

        // Gradient of EI(X) w.r.t X for LBFGS
        double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const {
            for (Eigen::Index i = 0; i < X_.cols(); i++) {
                k_star(i) = kernel.Kernel(X_.col(i), x);
            }
            const double kb = kernel.Kernel(x, x);
            const double muX = k_star.dot(alpha);

            imm = k_star;
            llt.solveInPlace(imm);

            const double variance = std::max(0.0, kb - k_star.dot(imm));
            const double sigmaX = std::sqrt(variance);

            // Zero variance check
            if (sigmaX < 1e-9) {
                grad.setZero(x.size());
                return 0.0;
            }

            const double Z = (muX - fx_best - exploration) / sigmaX;
            const double cdf_Z = norm_cdf(Z);
            const double pdf_Z = norm_pdf(Z);
            const double ei_value = (muX - fx_best - exploration) * cdf_Z + sigmaX * pdf_Z;

            for (Eigen::Index i = 0; i < X_.cols(); ++i) {
                const double sq_dist = (x - X_.col(i)).squaredNorm();
                const double weight = kernel.SpatialGradientWeight(sq_dist);
                grad_k_star.col(i) = -(x - X_.col(i)) * weight;
            }

            grad_mu.noalias() = grad_k_star * alpha;
            grad_sigma.noalias() = -(grad_k_star * imm) / sigmaX;

            grad.noalias() = -(cdf_Z * grad_mu + pdf_Z * grad_sigma);

            return -ei_value;
        }
    private:

        static double norm_pdf(const double x) {
            static constexpr double oneOv2pi = 0.39894228040143267794;
            return oneOv2pi * std::exp(-0.5 * x * x);
        }

        static double norm_cdf(const double x) {
            static constexpr double oneOvSqrt2 = 0.70710678118654752440;
            return 0.5 * (std::erf(x * oneOvSqrt2) + 1.0);
        }

        Matern52Kernel kernel;
        Eigen::MatrixXd X_;
        Eigen::VectorXd Y_;
        Eigen::LLT<Eigen::MatrixXd> llt;
        Eigen::VectorXd alpha;
        double fx_best;
        double exploration;

        mutable Eigen::VectorXd k_star;
        mutable Eigen::VectorXd imm;
        mutable Eigen::MatrixXd grad_k_star;
        mutable Eigen::VectorXd grad_mu;
        mutable Eigen::VectorXd grad_sigma;
    };

    [[nodiscard]] inline EI_Candidate OptimizeCandidateEIPoint(const ExpectedImprovement& eif, Eigen::VectorXd& candidateX) {
        LBFGSpp::LBFGSBParam<double> param_opt;
        param_opt.epsilon = 1e-6;
        param_opt.max_iterations = 100;

        LBFGSpp::LBFGSBSolver<double> solver(param_opt);

        // Lower and Upper bounds for the normalized space [0, 1]
        const Eigen::VectorXd lower_bounds = Eigen::VectorXd::Zero(candidateX.size());
        const Eigen::VectorXd upper_bounds = Eigen::VectorXd::Ones(candidateX.size());

        double fx_best_negative;

        try {
            solver.minimize(eif, candidateX, fx_best_negative, lower_bounds, upper_bounds);
        } catch (const std::exception& e) {
            fx_best_negative = eif(candidateX);
            std::cerr << "EI CAND OPT: " << e.what() << std::endl;
        }

        // Return the positive Expected Improvement
        return {candidateX, -fx_best_negative};
    }

};

#endif // AQFN_EXP_IMPROV_HPP