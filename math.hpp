/*
 * Copyright (c) 2020-present, Andrei Yaskovets
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdexcept>

/*
 * general-purpous math function
 */
namespace mlp
{
template<typename T>
constexpr auto pow(T x, int n) -> T
{
  if (n == 0)
    return T(1);
  if (n < 0)
    return T(1) / pow(x, -n);

  auto x_n = 1.0;
  for (; n > 0; n /= 2)
  {
    if (n & 1)
      x_n *= x;
    x *= x;
  }
  return x_n;
}

constexpr auto exp(double x) -> double
{
  if (x < -3.0 || 3.0 < x)
    return pow(exp(x / 2.0), 2);

  auto e_x = 1.0;
  auto last = 1.0;
  for (int n = 1; n < 20; ++n)
    e_x += (last *= x / static_cast<double>(n));
  return e_x;
}

constexpr auto ln(double x) -> double
{
  if (x < 0)
    throw std::invalid_argument("ln(negative)");

  auto l_x = double{};
  auto last = 2.0;
  for (int n = 0; n < 10; ++n)
  {
    const auto e_last = exp(last);
    last = (l_x += 2 * (x - e_last) / (x + e_last));
  }
  return l_x;
}
} // namespace mlp
