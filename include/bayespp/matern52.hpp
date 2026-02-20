#ifndef MATERN_HPP
#define MATERN_HPP
#pragma once

#include <Eigen/Dense>

#include <Eigen/Dense>
#include <cmath>

namespace bayespp {
    class Matern52Kernel {
    public:
        Matern52Kernel(const double sigma_sq, const double l_sq)
            : sigma_sq_(sigma_sq), l_sq_(l_sq), l_(std::sqrt(l_sq)) {}

        [[nodiscard]] Eigen::MatrixXd Covariance(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) const {
            Eigen::MatrixXd result(X.cols(), Y.cols());
            for (Eigen::Index i = 0; i < X.cols(); i++) {
                for (Eigen::Index j = 0; j < Y.cols(); j++) {
                    result(i, j) = Kernel(X.col(i), Y.col(j));
                }
            }
            return result;
        }

        [[nodiscard]] Eigen::MatrixXd Covariance(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, Eigen::MatrixXd& Sq_dist) const {
            Eigen::MatrixXd result(X.cols(), Y.cols());
            for (Eigen::Index i = 0; i < X.cols(); i++) {
                for (Eigen::Index j = 0; j < Y.cols(); j++) {
                    result(i, j) = Kernel(X.col(i), Y.col(j), Sq_dist(i, j));
                }
            }
            return result;
        }

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