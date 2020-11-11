# Compile time neural network

C++ constexpr-enabled multilayer perceptron. The implementation is relying on some C++17 quality of life features like fold expressions and C++14 extended constexpr support (loops and mutable variables in constexpr functions)

## Requirements

* C++17 compiler
* Default constexpr steps limit of your compiler may be exceeded when a significant number of training epochs is specified. Constexpr steps compiler flags can be used to get around this: _/constexpr:depth_ (MSVC), _-fconstexpr-steps_ (Clang)

## Example usage

The implementation relies heavily on operator overloading and functional programming patterns with the goal of simplifying the network code and the related math code. The code is living inside the __mlp__ namespace

* __Initializing a layer__

**Random weight initialization is to be implemented*

```c++
// l = layer of 3 neurons with 4 input connections
constexpr auto l = mlp::layer<3, 4>{mlp::act::Sigmoid, mlp::mat<double, 4, 3>{{...}}, mlp::vec<double, 4>{{...}}};
```

Math-related code consists of __mlp::mat__ and __mlp::vec__ types which are aliases for __std::array__

```c++
// m = 5x3 matrix
constexpr auto m = mlp::mat<int, 5, 3>{{ /* row-by-row initializer lists */ }};

// v = vector of 4 elements (column)
constexpr auto v = mlp::vec<char, 4>{{ /* column initializer list */ }};
```

Implemented activation functions are stored in __mlp::act__ enumeration

* __Layer composition__

Layers can be composed into a network using __operator+__

```c++
// l_i = input layer with 2 inputs and 4 neurons
constexpr auto l_i = mlp::layer<2, 4>{...};

// l_h = hidden layer with 4 inputs and 3 neurons
constexpr auto l_h = mlp::layer<4, 3>{...};

// l_o = output layer with 3 inputs and 1 nenuron
constexpr auto l_o = mlp::layer<3, 1>{...};

// network = perceptron with 2 inputs and 1 output
constexpr auto network = l_i + l_h + l_o;
```

* __Forwarding data__

Data stored in a __mlp::mat__ or __mlp::vec__ can be forwarded through the network using __operator>>__

```c++
// network = perceptron with 2 inputs and 3 outputs
constexpr auto network = mlp::layer<2, 4>{...} + mlp::layer<4, 3>{...};

// x_v = single set of input data
constexpr auto x_v = mlp::vec<float, 2>{...};

// x_m = five sets of input data stored as rows
constexpr auto x_m = mlp::mat<double, 5, 2>{...};

// y_v = mlp::vec<double, 3> which holds ouput layer values for x_v
constexpr auto y_v = x_v >> network;

// y_m = mlp::mat<double, 5, 3> which holds output layer values for x_m row-by-row
constexpr auto y_m = x_m >> network;
```

* __Fitting__

__fit__ function can be used to train the network. The function must be provided with the initial state of the network, training parameters __mlp::fitparms__, input values and desired output values

**Gradient descent optimizers are to be implemented*

```c++
// network = perceptron with 2 inputs and 1 output
constexpr auto network = mlp::layer<2, 4>{...} + mlp::layer<4, 3> + mlp::layer<3, 1>{...};

// parms = training environment with 1000 epochs, learning rate 0.1 and logistic loss function
constexpr auto parms = mlp::fitparms{1000, 0.1, mlp::lossf::LogLoss};

// x = training data
constexpr auto x = mlp::mat<double, 4, 2>{{{0, 0}, {0, 1}, {1, 0}, {1, 1}}};

// y = desired output
constexpr auto y = mlp::mat<double, 4, 1>{{0, 1, 1, 0}};

// network_fit = XOR perceptron trained using gradient descent and backpropagation
constexpr auto network_fit = mlp::fit(network, parms, x, y);
```

More detailed example can be found in [example.cpp](example.cpp)
