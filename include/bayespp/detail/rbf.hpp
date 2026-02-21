// Copyright (c) 2026 Jayawardane
// SPDX-License-Identifier: MIT
//
// This file is part of bayespp.
// See the LICENSE file in the project root for full license information.

#ifndef BAYESPP_RBF_HPP
#define BAYESPP_RBF_HPP
#pragma once

#include <Eigen/Dense>

namespace bayespp::detail {

    class RBFKernel {
    public:
        RBFKernel(const double sigma_sq, const double l_sq) : sigma_sq(sigma_sq), l_sq(l_sq) {}

        [[nodiscard]] double Kernel(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) const {
            const double dist_sq = (xi - xj).squaredNorm();
            return sigma_sq * std::exp(-dist_sq / (2.0 * l_sq));
        }

        [[nodiscard]] double Kernel(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj, double& sq_dist) const {
            const double dist_sq = (xi - xj).squaredNorm();
            sq_dist = dist_sq;
            return sigma_sq * std::exp(-dist_sq / (2.0 * l_sq));
        }

        [[nodiscard]] double get_length_scale_squared() const { return l_sq; }
        [[nodiscard]] double get_variance() const { return sigma_sq; }

    private:
        double sigma_sq;
        double l_sq;
    };

};
#endif // RBF_HPP