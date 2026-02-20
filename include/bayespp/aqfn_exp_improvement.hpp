#ifndef ATLAS_AQFN_EXP_IMPROV_HPP
#define ATLAS_AQFN_EXP_IMPROV_HPP
#pragma once

#include "rbf.hpp"
#include <LBFGSpp/LBFGSB.h>

#include "matern52.hpp"

namespace bayespp {
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
        explicit ExpectedImprovement(const Matern52Kernel& kernel, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,  const double fx_best, const double exploration = 0.01)
        : kernel(kernel), X_(X), Y_(Y), fx_best(fx_best), exploration(exploration) {
            auto K = kernel.Covariance(X, X);
            K.diagonal().array() += 1e-6;
            llt.compute(K);
            alpha = llt.solve(Y_);
        }

        // EI(X)
        double operator()(const Eigen::VectorXd& x) const {
            const Eigen::VectorXd k_star = kernel.Covariance(X_, x);
            const double kb = kernel.Kernel(x, x);

            const double muX = k_star.dot(alpha);
            const Eigen::VectorXd imm = llt.solve(k_star);
            const double variance = std::max(0.0, kb - k_star.dot(imm));
            const double sigmaX = std::sqrt(variance);

            if (sigmaX < 1e-9) {
                return 0.0;
            }

            const double Z = (muX - fx_best - exploration)/sigmaX;
            return (muX - fx_best - exploration) * norm_cdf(Z) + sigmaX * norm_pdf(Z);
        }

        // Gradient of EI(X) w.r.t X for LBFGS
        double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const {
            const Eigen::VectorXd k_star = kernel.Covariance(X_, x);
            const double kb = kernel.Kernel(x, x);

            const double muX = k_star.dot(alpha);
            const Eigen::VectorXd imm = llt.solve(k_star);

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

            const Eigen::Index D = x.size();
            const Eigen::Index N = X_.cols();

            Eigen::MatrixXd grad_k_star(D, N);

            for (Eigen::Index i = 0; i < N; ++i) {
                const double sq_dist = (x - X_.col(i)).squaredNorm();
                const double weight = kernel.SpatialGradientWeight(sq_dist);

                // Gradient of k(x, x_i) w.r.t x
                grad_k_star.col(i) = - (x - X_.col(i)) * weight;
            }

            Eigen::VectorXd grad_mu(D);
            grad_mu.noalias() = grad_k_star * alpha;

            Eigen::VectorXd grad_sigma(D);
            grad_sigma.noalias() = -(grad_k_star * imm) / sigmaX;

            grad = -(cdf_Z * grad_mu + pdf_Z * grad_sigma);

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
    };

    [[nodiscard]] inline EI_Candidate OptimizeCandidateEIPoint(const ExpectedImprovement& eif, Eigen::VectorXd& candidateX) {
        LBFGSpp::LBFGSBParam<double> param_opt;
        param_opt.epsilon = 1e-6;
        param_opt.max_iterations = 100;

        LBFGSpp::LBFGSBSolver<double> solver(param_opt);

        // Lower and Upper bounds for the normalized space [0, 1]
        Eigen::VectorXd lower_bounds = Eigen::VectorXd::Zero(candidateX.size());
        Eigen::VectorXd upper_bounds = Eigen::VectorXd::Ones(candidateX.size());

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

#endif //ATLAS_AQFN_EXP_IMPROV_HPP