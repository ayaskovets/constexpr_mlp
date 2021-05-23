/*
 * Copyright (c) 2020-present, Andrei Yaskovets
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <type_traits>

/*
 * vec definition
 */
namespace mlp
{
template<typename T, std::size_t M>
using vec = std::array<T, M>;
} // namespace mlp

/*
 * vec functional
 */
namespace mlp
{
template<typename F, typename A, std::size_t M>
constexpr auto fmap(F&& f, const vec<A, M>& a) -> vec<std::invoke_result_t<F, A>, M>
{
  auto b = vec<std::invoke_result_t<F, A>, M>{};
  for (std::size_t i = 0; i < M; ++i)
    b[i] = f(a[i]);
  return b;
}

template<typename F, typename A, typename B, std::size_t M>
constexpr auto zip(F&& f, const vec<A, M>& a, const vec<B, M>& b) -> vec<std::invoke_result_t<F, A, B>, M>
{
  auto c = vec<std::invoke_result_t<F, A, B>, M>{};
  for (std::size_t i = 0; i < M; ++i)
    c[i] = f(a[i], b[i]);
  return c;
}

template<typename F, typename A, typename B, std::size_t M>
constexpr auto fold(F&& f, const A& z, const vec<B, M>& x) -> std::conditional_t<M == 0, A, std::invoke_result_t<F, A, B>>
{
  if constexpr (M == 0)
    return z;
  else
  {
    auto r = std::invoke_result_t<F, A, B>{f(z, x[0])};
    for (std::size_t i = 1; i < M; ++i)
      r = f(r, x[i]);
    return r;
  }
}
} // namespace mlp

/*
 * vec operations
 */
namespace mlp
{
template<typename A, typename B, std::size_t M>
constexpr auto operator+(const vec<A, M>& a, const vec<B, M>& b) -> vec<decltype(A{} + B{}), M>
{
  return zip([](A a_m, B b_m){ return a_m + b_m; }, a, b);
}

template<typename A, typename B, std::size_t M>
constexpr auto operator-(const vec<A, M>& a, const vec<B, M>& b) -> vec<decltype(A{} + B{}), M>
{
  return zip([](A a_m, B b_m){ return a_m - b_m; }, a, b);
}

template<typename A, typename B, std::size_t M>
constexpr auto operator*(const vec<A, M>& a, B b) -> vec<decltype(A{} * B{}), M>
{
  return fmap([b](A a_m){ return a_m * b; }, a);
}
} // namespace mlp

/*
 * mat
 */
namespace mlp
{
template<typename T, std::size_t M, std::size_t N>
using mat = vec<vec<T, N>, M>;
} // namespace mlp

/*
 * mat functional
 */
namespace mlp
{
template<typename F, typename A, std::size_t M, std::size_t N>
constexpr auto fmap(F&& f, const mat<A, M, N>& a) -> mat<std::invoke_result_t<F, A>, M, N>
{
  auto b = mat<std::invoke_result_t<F, A>, M, N>{};
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j)
      b[i][j] = f(a[i][j]);
  return b;
}

template<typename F, typename A, typename B, std::size_t M, std::size_t N>
constexpr auto zip(F&& f, const mat<A, M, N>& a, const mat<B, M, N>& b) -> mat<std::invoke_result_t<F, A, B>, M, N>
{
  auto c = mat<std::invoke_result_t<F, A, B>, M, N>{};
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j)
      c[i][j] = f(a[i][j], b[i][j]);
  return c;
}
} // namespace mlp

/*
 * mat operations
 */
namespace mlp
{
template<typename A, typename B, std::size_t M, std::size_t N>
constexpr auto operator+(const mat<A, M, N>& a, const mat<B, M, N>& b) -> mat<decltype(A{} + B{}), M, N>
{
  return zip([](A a_m, B b_m){ return a_m + b_m; }, a, b);
}

template<typename A, typename B, std::size_t M, std::size_t N>
constexpr auto operator-(const mat<A, M, N>& a, const mat<B, M, N>& b) -> mat<decltype(A{} + B{}), M, N>
{
  return zip([](A a_m, B b_m){ return a_m - b_m; }, a, b);
}

template<typename A, typename B, std::size_t M, std::size_t N, std::size_t P>
constexpr auto operator*(const mat<A, M, N>& a, const mat<B, N, P>& b) -> mat<decltype(A{} * B{}), M, P>
{
  auto c = mat<decltype(A{} * B{}), M, P>{};
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t p = 0; p < P; ++p)
      for (std::size_t j = 0; j < N; ++j)
        c[i][p] = c[i][p] + a[i][j] * b[j][p];
  return c;
}

template<typename A, typename B, std::size_t M, std::size_t N>
constexpr auto operator*(const mat<A, M, N>& a, B b) -> mat<decltype(A{} * B{}), M, N>
{
  return fmap([b](A a_m){ return a_m * b; }, a);
}
} // namespace mlp

/*
 * cross-type operations
 */
namespace mlp
{
template<typename A, typename B, std::size_t M, std::size_t N>
constexpr auto operator*(const mat<A, M, N>& a, const vec<B, N>& b) -> vec<decltype(A{} * B{}), M>
{
  auto c = vec<decltype(A{} * B{}), M>{};
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j)
      c[i] = c[i] + a[i][j] * b[j];
  return c;
}

template<typename A, typename B, std::size_t M, std::size_t N>
constexpr auto operator*(const vec<A, M>& a, const mat<B, 1, N>& b) -> mat<decltype(A{} * B{}), M, N>
{
  auto c = mat<decltype(A{} * B{}), M, N>{};
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j)
      c[i][j] = a[i] * b[0][j];
  return c;
}

template<typename A, typename B, std::size_t M>
constexpr auto operator+(const vec<A, M>& a, const mat<B, M, 1>& b) -> vec<decltype(A{} + B{}), M>
{
  auto c = vec<decltype(A{} + B{}), M>{};
  for (std::size_t i = 0; i < M; ++i)
    c[i] = a[i] + b[i][0];
  return c;
}

template<typename A, typename B, std::size_t M>
constexpr auto operator-(const vec<A, M>& a, const mat<B, M, 1>& b) -> vec<decltype(A{} + B{}), M>
{
  auto c = vec<decltype(A{} + B{}), M>{};
  for (std::size_t i = 0; i < M; ++i)
    c[i] = a[i] - b[i][0];
  return c;
}

template<typename A, typename B, std::size_t M>
constexpr auto operator+(const mat<A, M, 1>& a, const vec<B, M>& b) -> vec<decltype(A{} + B{}), M>
{
  return b + a;
}

template<typename A, typename B, std::size_t M>
constexpr auto operator-(const mat<A, M, 1>& a, const vec<B, M>& b) -> vec<decltype(A{} + B{}), M>
{
  return b - a;
}

template<typename T, std::size_t M>
constexpr auto transpose(const vec<T, M>& v) -> mat<T, 1, M>
{
  return {{v}};
}

template<typename T, std::size_t M, std::size_t N>
constexpr auto transpose(const mat<T, M, N>& a) -> mat<T, N, M>
{
  auto a_t = mat<T, N, M>{};
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j)
      a_t[j][i] = a[i][j];
  return a_t;
}
} // namespace mlp
