#ifndef PARAM_SPACE_HPP
#define PARAM_SPACE_HPP
#pragma once

#include <cmath>
#include <vector>
#include <Eigen/Dense>

namespace bayespp {

    class ParameterSpace {
    public:

        ParameterSpace() = default;

        enum class Type {
            Real,
            RealLog,
            RealFixed,
            Integer,
            IntegerFixed,
            Option,
            OptionFixed
        };

        struct Parameter {
            Type type;
            double min;
            double max;
            double fixed_val;
            int num_options;

            static Parameter MakeReal(const double min, const double max) { return {Type::Real, min, max, 0.0, 0}; }
            static Parameter MakeRealLog(const double min, const double max) { return {Type::RealLog, std::max(min, 0.0), max, 0.0, 0}; }
            static Parameter MakeRealFixed(const double val) { return {Type::RealFixed, 0.0, 0.0, val, 0}; }
            static Parameter MakeInt(const double min, const double max) { return {Type::Integer, min, max, 0.0, 0}; }
            static Parameter MakeIntFixed(const double val) { return {Type::IntegerFixed, 0.0, 0.0, val, 0}; }
            static Parameter MakeOption(const int num_options) { return {Type::Option, 0.0, 0.0, 0.0, num_options}; }
            static Parameter MakeOptionFixed(const double val) { return {Type::OptionFixed, 0.0, 0.0, val, 0}; }
        };

        void AddRealParameter(const double min, const double max) { add_parameter(Parameter::MakeReal(min, max)); }
        void AddLogRealParameter(const double min, const double max) { add_parameter(Parameter::MakeRealLog(min, max)); }
        void AddFixedRealParameter(const double value) { add_parameter(Parameter::MakeRealFixed(value)); }
        void AddIntegerParameter(const double min, const double max) { add_parameter(Parameter::MakeInt(min, max)); }
        void AddFixedIntegerParameter(const double value) { add_parameter(Parameter::MakeIntFixed(value)); }
        void AddOptionParameter(const int num_options) { add_parameter(Parameter::MakeOption(num_options)); }
        void AddFixedOptionParameter(const double value) { add_parameter(Parameter::MakeOptionFixed(value)); }

        [[nodiscard]] size_t num_parameters() const { return parameters.size(); }

        [[nodiscard]] size_t num_normalized_params() const { return norm_params; }

        // The inverse transformation of vector
        [[nodiscard]] std::vector<double> inverse_transform(const std::vector<double>& normalized) const {
            std::vector<double> result;
            result.reserve(parameters.size());

            size_t norm_idx = 0;

            for (const auto&[type, min, max, fixed_val, num_options] : parameters) {
                switch (type) {
                    case Type::Real:
                        result.push_back(min + normalized[norm_idx++] * (max - min));
                        break;

                    case Type::RealLog:
                        result.push_back(min * std::pow(max / min, normalized[norm_idx++]));
                        break;

                    case Type::RealFixed:
                        result.push_back(fixed_val);
                        break;

                    case Type::Integer:
                        result.push_back(std::round(min + normalized[norm_idx++] * (max - min)));
                        break;

                    case Type::IntegerFixed:
                        result.push_back(fixed_val);
                        break;

                    case Type::Option: {
                        int best_idx = 0;
                        double best_val = -1.0;
                        // Consume 'num_options' values and find the argmax
                        for (int i = 0; i < num_options; ++i) {
                            if (normalized[norm_idx] > best_val) {
                                best_val = normalized[norm_idx];
                                best_idx = i;
                            }
                            norm_idx++;
                        }
                        result.push_back(static_cast<double>(best_idx));
                        break;
                    }

                    case Type::OptionFixed:
                        result.push_back(fixed_val);
                        break;
                }
            }

            return result;
        }

        // The inverse transformation eigen vector
        template<typename Derived>
        [[nodiscard]] std::vector<double> inverse_transform(const Eigen::MatrixBase<Derived>& normalized) const {
            EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);

            std::vector<double> result;
            result.reserve(parameters.size());

            long long norm_idx = 0;

            for (const auto&[type, min, max, fixed_val, num_options] : parameters) {
                switch (type) {
                    case Type::Real:
                        result.push_back(min + normalized(norm_idx) * (max - min));
                        norm_idx++;
                        break;

                    case Type::RealLog:
                        result.push_back(min * std::pow(max / min, normalized(norm_idx)));
                        norm_idx++;
                        break;

                    case Type::RealFixed:
                        result.push_back(fixed_val);
                        break;

                    // std::min(std::floor(x*(max−min+1))+min, max) has a uniform distribution
                    case Type::Integer:
                        result.push_back(std::round(min + normalized(norm_idx) * (max - min)));
                        norm_idx++;
                        break;

                    case Type::IntegerFixed:
                        result.push_back(fixed_val);
                        break;

                    case Type::Option: {
                        int best_idx = 0;
                        normalized.segment(norm_idx, num_options).maxCoeff(&best_idx);
                        result.push_back(static_cast<double>(best_idx));
                        break;
                    }

                    case Type::OptionFixed:
                        result.push_back(fixed_val);
                        break;
                }
            }

            return result;
        }

    private:
        void add_parameter(const Parameter& p) {
            if (p.type == Type::Real || p.type == Type::RealLog || p.type == Type::Integer) {
                norm_params++;
            }
            else if (p.type == Type::Option) {
                norm_params += p.num_options;
            }
            parameters.push_back(p);
        }

        std::vector<Parameter> parameters;
        size_t norm_params = 0;
    };

};
#endif // PARAM_SPACE_HPP