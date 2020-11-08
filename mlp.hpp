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
