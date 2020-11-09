#include "mlp.hpp"

#include <iomanip>
#include <iostream>

int main()
{
  using namespace mlp;

  // net layers
  constexpr auto i = layer<2, 4>{act::ReLU, {{{.1, .2}, {0.3, .4}, {.5, -0.6}}}, {}};
  constexpr auto h = layer<4, 3>{act::ReLU, {{{.1, .2, .3}, {.4, .5, .2}}}, {}};
  constexpr auto o = layer<3, 1>{act::Sigmoid, {{.1, .2, .3}}, {}};

  // initial net
  constexpr auto net1 = i + h + o;

  // XOR truth table
  constexpr auto x_train = mat<double, 4, 2>{{{0, 0}, {0, 1}, {1, 0}, {1, 1}}};
  constexpr auto y_train = mat<double, 4, 1>{{0, 1, 1, 0}};

  // initial prediction
  {
    constexpr auto y_pred = x_train >> net1;

    std::cout << "initial predictions: \n";
    for (std::size_t i = 0; i < y_pred.size(); ++i)
      std::cout << "\tnet(" <<
        std::setw(2) << x_train[i][0] << "," <<
        std::setw(2) << x_train[i][1] << ")=" << y_pred[i][0] << '\n';
  }

  // fitted network
  constexpr auto net2 = fit(net1, fitparms{500, 0.05, lossf::LogLoss}, x_train, y_train);

  // trained prediction
  {
    // testing values
    constexpr auto x_test = mat<double, 8, 2>{{
      {0, 0}, {0, 1}, {1, 0}, {1, 1},
      {1, -1}, {0, 2}, {3, 0}, {15, 15}
    }};

    constexpr auto y_pred = x_test >> net2;

    std::cout << "trained predictions: \n";
    for (std::size_t i = 0; i < y_pred.size(); ++i)
      std::cout << "\tnet(" <<
        std::setw(2) << x_test[i][0] << "," <<
        std::setw(2) << x_test[i][1] << ")=" << y_pred[i][0] << '\n';
  }
}
