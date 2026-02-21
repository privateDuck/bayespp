// Copyright (c) 2026 Jayawardane
// SPDX-License-Identifier: MIT
//
// This file is part of bayespp.
// See the LICENSE file in the project root for full license information.

#ifndef BAYESPP_MATERN_HPP
#define BAYESPP_MATERN_HPP
#pragma once

#include <Eigen/Dense>

namespace bayespp::detail {

    class Matern52Kernel {
    public:
        Matern52Kernel(const double sigma_sq, const double l_sq)
            : sigma_sq_(sigma_sq), l_sq_(l_sq), l_(std::sqrt(l_sq)) {}

        [[nodiscard]] double Kernel(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) const {
            const double dist = (xi - xj).norm();
            const double sqrt5_d_l = SQRT5 * dist / l_;
            return sigma_sq_ * (1.0 + sqrt5_d_l + (5.0 * dist * dist) / (3.0 * l_sq_)) * std::exp(-sqrt5_d_l);
        }

        [[nodiscard]] double Kernel(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj, double& sq_dist) const {
            const double dist_sq = (xi - xj).squaredNorm();
            sq_dist = dist_sq;
            const double dist = std::sqrt(dist_sq);
            const double sqrt5_d_l = SQRT5 * dist / l_;
            return sigma_sq_ * (1.0 + sqrt5_d_l + (5.0 * dist_sq) / (3.0 * l_sq_)) * std::exp(-sqrt5_d_l);
        }

        // The scalar multiplier for the X gradient
        [[nodiscard]] double SpatialGradientWeight(const double sq_dist) const {
            const double dist = std::sqrt(sq_dist);
            const double sqrt5_d_l = SQRT5 * dist / l_;
            return (5.0 * sigma_sq_ / (3.0 * l_sq_)) * (1.0 + sqrt5_d_l) * std::exp(-sqrt5_d_l);
        }

        // The scalar derivative w.r.t log(l)
        [[nodiscard]] double LGradientWeight(const double sq_dist) const {
            const double dist = std::sqrt(sq_dist);
            const double sqrt5_d_l = SQRT5 * dist / l_;
            return (5.0 * sigma_sq_ * sq_dist / (3.0 * l_sq_)) * (1.0 + sqrt5_d_l) * std::exp(-sqrt5_d_l);
        }

        [[nodiscard]] double get_length_scale_squared() const { return l_sq_; }
        [[nodiscard]] double get_variance() const { return sigma_sq_; }

    private:
        double sigma_sq_;
        double l_sq_;
        double l_;
        static constexpr double SQRT5 = 2.2360679774997896964091736687313;
    };
};

#endif //MATERN_HPP