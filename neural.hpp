/*
 * Copyright (c) 2020-present, Andrei Yaskovets
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "math.hpp"
#include "matrix.hpp"

/*
 * activation definition
 */
namespace mlp
{
enum class act : int
{
  Linear,
  ReLU,
  Sigmoid,
  Tanh
};

template<act A>
constexpr auto activation(double x) -> double
{
  if constexpr (A == act::Linear)
    return x;
  if constexpr (A == act::ReLU)
    return std::max(0.0, x);
  if constexpr (A == act::Sigmoid)
    return 1.0 / (1.0 + exp(-x));
  if constexpr (A == act::Tanh)
    return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
}

template<std::size_t M>
constexpr auto activation(act f, const vec<double, M>& x) -> vec<double, M>
{
  switch (f)
  {
  case act::Linear:
    return fmap(activation<act::Linear>, x);
  case act::ReLU:
    return fmap(activation<act::ReLU>, x);
  case act::Sigmoid:
    return fmap(activation<act::Sigmoid>, x);
  case act::Tanh:
    return fmap(activation<act::Tanh>, x);
  }
}
} // namespace mlp

/*
 * activation derivative
 */
namespace mlp
{
template<act A>
constexpr auto derivative(double x) -> double
{
  if constexpr (A == act::Linear)
    return 1;
  if constexpr (A == act::ReLU)
    return x < 0.0 ? 0.0 : 1.0;
  if constexpr (A == act::Sigmoid)
    return [](double a){ return a * (1.0 - a); }(activation<act::Sigmoid>(x));
  if constexpr (A == act::Tanh)
    return 1.0 - pow(activation<act::Tanh>(x), 2);
}

template<std::size_t M>
constexpr auto derivative(act f, const vec<double, M>& x) -> vec<double, M>
{
  switch (f)
  {
  case act::Linear:
    return fmap(derivative<act::Linear>, x);
  case act::ReLU:
    return fmap(derivative<act::ReLU>, x);
  case act::Sigmoid:
    return fmap(derivative<act::Sigmoid>, x);
  case act::Tanh:
    return fmap(derivative<act::Tanh>, x);
  }
}
} // namespace mlp

/*
 * loss definition
 */
namespace mlp
{
enum class lossf : int
{
  MSE,
  LogLoss
};

template<lossf L>
constexpr auto loss(double y_real, double y_pred) -> double
{
  if constexpr (L == lossf::MSE)
    return pow(y_real - y_pred, 2);
  if constexpr (L == lossf::LogLoss)
    return y_real * ln(y_pred) + (1.0 - y_real) * ln(1.0 - y_pred);
}

template<std::size_t M>
constexpr auto loss(lossf f, const vec<double, M>& y_real, const vec<double, M>& y_pred) -> double
{
  switch (f)
  {
  case lossf::MSE:
    return fold(std::plus{}, 0.0, zip(loss<lossf::MSE>, y_real, y_pred)) / static_cast<double>(M);
  case lossf::LogLoss:
    return fold(std::plus{}, 0.0, zip(loss<lossf::LogLoss>, y_real, y_pred)) / -static_cast<double>(M);
  }
}

template<std::size_t M, std::size_t N>
constexpr auto loss(lossf f, const mat<double, M, N>& y_real, const mat<double, M, N>& y_pred) -> double
{
  return fold(std::plus{}, 0.0, zip([f](const vec<double, N>& y_r, const vec<double, N>& y_p){
    return loss(f, y_r, y_p); }, y_real, y_pred)) / static_cast<double>(N);
}
} // namespace mlp

/*
 * loss derivative
 */
namespace mlp
{
template<lossf L>
constexpr auto derivative(double y_real, double y_pred) -> double
{
  if constexpr (L == lossf::MSE)
    return -2 * (y_real - y_pred);
  if constexpr (L == lossf::LogLoss)
    return (y_pred - y_real) / (y_pred * (1.0 - y_pred));
}

template<std::size_t M>
constexpr auto derivative(lossf f, const vec<double, M>& y_real, const vec<double, M>& y_pred) -> vec<double, M>
{
  switch (f)
  {
  case lossf::MSE:
    return zip(derivative<lossf::MSE>, y_real, y_pred);
  case lossf::LogLoss:
    return zip(derivative<lossf::LogLoss>, y_real, y_pred);
  }
}
} // namespace mlp
