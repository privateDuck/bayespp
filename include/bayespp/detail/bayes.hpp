// Copyright (c) 2026 Jayawardane
// SPDX-License-Identifier: MIT
//
// This file is part of bayespp.
// See the LICENSE file in the project root for full license information.

#ifndef BAYESPP_BAYES_HPP
#define BAYESPP_BAYES_HPP
#pragma once

#include <random>
#include <queue>
#include "aqfn_exp_improvement.hpp"
#include "nlml_opt.hpp"
#include "param_space.hpp"

namespace bayespp {

    struct BayesParameters {
        double exploration_parameter = 0.01;
        int n_evaluations = 100;
        int n_initial_points = 5;
        int n_acq_samples = 1000;
        int n_acq_restarts = 3;
        bool enable_kernel_multistart = false;
    };

    class BayesOptimizer {
    public:
        BayesOptimizer(const ParameterSpace& params, const BayesParameters& optimizer_parameters)
        : param_space_(params),
            generator_(std::random_device{}()),
            distribution_(0.0, 1.0),
            uniform_buf_(params.num_normalized_params()),
            opt_params_(optimizer_parameters),
            num_normalized_params(params.num_normalized_params())
        {

        }

        template <typename Foo>
        double Maximize(Foo& func, std::vector<double>& optim_param) {
            Eigen::MatrixXd Xs(num_normalized_params, opt_params_.n_evaluations + opt_params_.n_initial_points);
            Eigen::VectorXd Ys(opt_params_.n_evaluations + opt_params_.n_initial_points);
            Eigen::Index eval_points = 0;

            for (int i = 0; i < opt_params_.n_initial_points; i++) {
                Xs.col(eval_points) = get_next_rand();
                auto inv_tr = param_space_.inverse_transform(Xs.col(eval_points));
                Ys(eval_points) = func(inv_tr);
                eval_points++;
            }

            for (int i = 0; i < opt_params_.n_evaluations; i++) {
                auto current_X = Xs.leftCols(eval_points);
                auto current_Y = Ys.head(eval_points);
                double meanY = current_Y.mean();
                double stdY = std::sqrt((current_Y.array() - meanY).square().mean());
                auto Y_std = (current_Y.array() - meanY) / stdY;

                auto defaultKern =
                    opt_params_.enable_kernel_multistart ?
                detail::ComputeOptimalKernelMultiStart(current_X, Y_std)
                : detail::ComputeOptimalKernel(current_X, Y_std);

                const double best = Y_std.maxCoeff();
                detail::ExpectedImprovement ei(defaultKern, current_X, Y_std, best, opt_params_.exploration_parameter);

                // Generate candidate points with ei
                for (int c = 0; c < opt_params_.n_acq_samples; c++) {
                    get_next_rand();
                    const double ei_x = ei(uniform_buf_);
                    add_ei_opt_point(uniform_buf_, ei_x);
                }

                // Optimize ei on the best candidate points
                Xs.col(eval_points) = candidate_min_heap_.top().candidate;
                double y_best = std::numeric_limits<double>::min();
                while (!candidate_min_heap_.empty()) {
                    auto [candidate, y] = candidate_min_heap_.top();
                    detail::EI_Candidate optimized_candidate = OptimizeCandidateEIPoint(ei, candidate);
                    if (optimized_candidate.y > y_best) {
                        y_best = optimized_candidate.y;
                        Xs.col(eval_points) = candidate;
                    }
                    candidate_min_heap_.pop();
                }

                // Now we actually evaluate function at x_best
                auto inv_tr = param_space_.inverse_transform(Xs.col(eval_points));
                y_best = func(inv_tr);

                // Xs.col(eval_points) = x_best;
                Ys(eval_points) = y_best;
                eval_points++;
            }

            Eigen::Index optim_index;
            const double optim_fx = Ys.maxCoeff(&optim_index);
            optim_param = param_space_.inverse_transform(Xs.col(optim_index));
            return optim_fx;
        }

    private:

        const Eigen::VectorXd& get_next_rand() {
            double* data_ptr = uniform_buf_.data();
            for(int i = 0; i < num_normalized_params; ++i) {
                data_ptr[i] = distribution_(generator_);
            }
            return uniform_buf_;
        }

        void add_ei_opt_point(const Eigen::VectorXd& candidate, const double y) {
            if (candidate_min_heap_.size() < opt_params_.n_acq_restarts) {
                candidate_min_heap_.push({candidate, y});
            } else if (y > candidate_min_heap_.top().y) {
                candidate_min_heap_.pop();
                candidate_min_heap_.push({candidate, y});
            }
        }

        ParameterSpace param_space_;
        std::mt19937 generator_;
        std::uniform_real_distribution<double> distribution_;
        Eigen::VectorXd uniform_buf_;
        BayesParameters opt_params_;
        std::priority_queue<detail::EI_Candidate, std::vector<detail::EI_Candidate>, std::greater<>> candidate_min_heap_;
        size_t num_normalized_params;
    };

};
#endif // BAYES_HPP