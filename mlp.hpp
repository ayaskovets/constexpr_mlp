/*
 * Copyright (c) 2020-present, Andrei Yaskovets
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "neural.hpp"

#include <type_traits>
#include <tuple>

/*
 * layer definition
 */
namespace mlp
{
template<std::size_t I, std::size_t O>
struct layer
{
  act a;
  mat<double, O, I> w;
  vec<double, O> b;
};
} // namespace mlp

/*
 * layer operations
 */
namespace mlp
{
template<std::size_t I, std::size_t O>
constexpr auto operator>>(const vec<double, I>& x, const layer<I, O>& l) -> vec<double, O>
{
  return activation(l.a, l.w * x + l.b);
}

template<std::size_t I, std::size_t O, std::size_t N>
constexpr auto operator>>(const mat<double, N, I>& x, const layer<I, O>& l) -> mat<double, N, O>
{
    return fmap([&l](const vec<double, I>& x_i){ return x_i >> l; }, x);
}
} // namespace mlp

/*
 * mlp definition
 */
namespace mlp
{
template<typename... Ls>
using mlp = std::tuple<Ls...>;
} // namespace mlp

/*
 * mlp composition operations
 */
namespace mlp
{
template<std::size_t I, std::size_t N, std::size_t O>
constexpr auto operator+(const layer<I, N>& li, const layer<N, O>& lo) -> mlp<layer<I, N>, layer<N, O>>
{
  return {li, lo};
}

template<typename... Ls, std::size_t I, std::size_t N>
constexpr auto operator+(const mlp<Ls...>& net, const layer<I, N>& l) -> mlp<Ls..., layer<I, N>>
{
  static_assert(sizeof(decltype(std::get<sizeof...(Ls) - 1>(net) + l)));
  return std::tuple_cat(net, std::make_tuple(l));
}
} // namespace mlp

/*
 * mlp data forwarding operations
 */
namespace mlp
{
template<std::size_t I, std::size_t O, typename... Ls>
constexpr auto operator>>(const vec<double, I>& x, const mlp<layer<I, O>, Ls...>& net)
{
  return std::apply([&x](const auto&... ls){ return (x >> ... >> ls); }, net);
}

template<std::size_t I, std::size_t O, std::size_t N, typename... Ls>
constexpr auto operator>>(const mat<double, N, I>& x, const mlp<layer<I, O>, Ls...>& net)
{
  return std::apply([&x](const auto&... ls){ return (x >> ... >> ls); }, net);
}
} // namespace mlp

/*
 * mlp training
 */
namespace mlp
{
struct fitparms
{
  std::size_t epochs;
  double rate;
  lossf loss;
};

template<std::size_t L = 0, std::size_t I, std::size_t O, typename... Ls>
constexpr auto backpropagate(mlp<Ls...>& net, const fitparms& par, const vec<double, I>& x, const vec<double, O>& y)
{
  static_assert(L < sizeof...(Ls));

  auto& l = std::get<L>(net);
  const auto z = l.w * x + l.b;
  const auto a = activation(l.a, z);

  auto delta = derivative(l.a, z);
  if constexpr (L == sizeof...(Ls) - 1)
    delta = zip(std::multiplies{}, delta, derivative(par.loss, y, a));
  else
  {
    const auto w_next = transpose(std::get<L + 1>(net).w);
    delta = zip(std::multiplies{}, delta, w_next * backpropagate<L + 1>(net, par, a, y));
  }

  l.w = l.w - delta * transpose(x) * par.rate;
  l.b = l.b - delta * par.rate;

  return delta;
}

template<std::size_t N, std::size_t I, std::size_t O, typename... Ls>
constexpr auto fit(const mlp<Ls...>& net, const fitparms& par, const mat<double, N, I>& x, const mat<double, N, O>& y) -> mlp<Ls...>
{
  auto fnet = net;
  for (std::size_t epoch = 1; epoch <= par.epochs; ++epoch)
    for (std::size_t n = 0; n < N; ++n)
      backpropagate(fnet, par, x[n], y[n]);
  return fnet;
}
} // namespace mlp
