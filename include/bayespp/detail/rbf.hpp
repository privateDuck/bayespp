#ifndef RBF_HPP
#define RBF_HPP
#pragma once

#include <Eigen/Dense>

namespace bayespp::detail {

    class RBFKernel {
    public:
        RBFKernel(const double sigma_sq, const double l_sq) : sigma_sq(sigma_sq), l_sq(l_sq) {}

        [[nodiscard]] Eigen::MatrixXd Covariance(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) const {
            Eigen::MatrixXd result(X.cols(), Y.cols());
            for (Eigen::Index i = 0; i < X.cols(); i++) {
                for (Eigen::Index j = 0; j < Y.cols(); j++) {
                    result(i,j) = Kernel(X.col(i), Y.col(j));
                }
            }
            return result;
        }

        [[nodiscard]] Eigen::MatrixXd Covariance(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, Eigen::MatrixXd& Sq_dist) const {
            Eigen::MatrixXd result(X.cols(), Y.cols());
            for (Eigen::Index i = 0; i < X.cols(); i++) {
                for (Eigen::Index j = 0; j < Y.cols(); j++) {
                    result(i,j) = Kernel(X.col(i), Y.col(j), Sq_dist(i,j));
                }
            }
            return result;
        }

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