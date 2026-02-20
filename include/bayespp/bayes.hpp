#ifndef ATLAS_BAYES_HPP
#define ATLAS_BAYES_HPP
#pragma once

#include <random>
#include <queue>
#include "aqfn_exp_improvement.hpp"
#include "nlml_opt.hpp"
#include "param_space.hpp"

namespace bayespp {

    struct BayesParameters {
        int max_iterations = 100;
        int initial_guesses = 5;
        int max_ei_candidates = 1000;
        int max_ei_opt_candidates = 3;
        double exploration_parameter = 0.01;
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
            Eigen::MatrixXd Xs(num_normalized_params, opt_params_.max_iterations + opt_params_.initial_guesses);
            Eigen::VectorXd Ys(opt_params_.max_iterations + opt_params_.initial_guesses);
            Eigen::Index eval_points = 0;

            for (int i = 0; i < opt_params_.initial_guesses; i++) {
                Xs.col(eval_points) = get_next_rand();
                auto inv_tr = param_space_.inverse_transform(Xs.col(eval_points));
                Ys(eval_points) = func(inv_tr);
                eval_points++;
            }

            double initialSignalVar = (Ys.head(eval_points).array() - Ys.head(eval_points).mean()).square().mean();
            double initialLScale = 2.0;
            auto defaultKern = RBFKernel(initialSignalVar, initialLScale * initialLScale);

            for (int i = 0; i < opt_params_.max_iterations; i++) {
                auto current_X = Xs.leftCols(eval_points);
                auto current_Y = Ys.head(eval_points);
                double meanY = current_Y.mean();
                double stdY = std::sqrt((current_Y.array() - meanY).square().mean());
                Eigen::VectorXd Y_std = (current_Y.array() - meanY) / stdY;
                defaultKern = ComputeOptimalRBFKernel(current_X, Y_std);

                const double best = Y_std.maxCoeff();
                ExpectedImprovement ei(defaultKern, current_X, Y_std, best, opt_params_.exploration_parameter);

                // Generate candidate points with ei
                for (int c = 0; c < opt_params_.max_ei_candidates; c++) {
                    get_next_rand();
                    const double ei_x = ei(uniform_buf_);
                    add_ei_opt_point(uniform_buf_, ei_x);
                }

                // Optimize ei on the best candidate points
                Eigen::VectorXd x_best = candidate_min_heap_.top().candidate;
                double y_best = std::numeric_limits<double>::min();
                while (!candidate_min_heap_.empty()) {
                    auto [candidate, y] = candidate_min_heap_.top();
                    EI_Candidate optimized_candidate = OptimizeCandidateEIPoint(ei, candidate);
                    if (optimized_candidate.y > y_best) {
                        y_best = optimized_candidate.y;
                        x_best = candidate;
                    }
                    candidate_min_heap_.pop();
                }

                // Now we actually evaluate function at x_best
                auto inv_tr = param_space_.inverse_transform(x_best);
                y_best = func(inv_tr);

                // append new x_best and Foo(x_best) to Xsample and Ysample
                Xs.col(eval_points) = x_best;
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
            if (candidate_min_heap_.size() < opt_params_.max_ei_opt_candidates) {
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
        std::priority_queue<EI_Candidate, std::vector<EI_Candidate>, std::greater<EI_Candidate>> candidate_min_heap_;
        int num_normalized_params;
    };

};
#endif //ATLAS_BAYES_HPP