"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """Multiplies two numbers

    Args:
    ----
        x: float number
        y: float number

    Returns:
    -------
        x times y

    """
    return x * y


# - id
def id(x: float) -> float:
    """Returns input unchanched

    Args:
    ----
        x: Any number

    Returns:
    -------
        Input unchanged

    """
    return x


# - add
def add(x: float, y: float) -> float:
    """Adds two numbers

    Args:
    ----
        x: float number
        y: float number

    Returns:
    -------
        x plus y

    """
    return x + y


# - neg
def neg(x: float) -> float:
    """Negates a number

    Args:
    ----
        x: float number

    Returns:
    -------
        -x

    """
    return -1 * x


# - lt
def lt(x: float, y: float) -> bool:
    """Checks if x < y

    Args:
    ----
        x: float number
        y: float number

    Returns:
    -------
        True if x<y, False otherwise

    """
    return x < y


# - eq
def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal

    Args:
    ----
        x: float number
        y: float number

    Returns:
    -------
        True if x=y, false otherwise

    """
    return x == y


# - max
def max(x: float, y: float) -> float:
    """Returns the largest of two numbers

    Args:
    ----
        x: float number
        y: float number

    Returns:
    -------
        Greater number

    """
    if x >= y:
        return x
    return y


# - is_close
def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value

    Args:
    ----
        x: float number
        y: float number

    Returns:
    -------
        True if they are close, False otherwise

    """
    eps = 1e-2
    return (x - y < eps) and (x - y > neg(eps))


# - sigmoid
def sigmoid(x: float) -> float:
    """Calculates sigmoid value at x

    Args:
    ----
        x: float number

    Returns:
    -------
        f(x)

    """
    return 1 / (1 + math.exp(-x))


# - relu
def relu(x: float) -> float:
    """Calculates relu value at x

    Args:
    ----
        x: float number

    Returns:
    -------
        f(x)

    """
    if x < 0:
        return 0
    return x


# - log
def log(x: float) -> float:
    """Calculates log of x

    Args:
    ----
        x: float number

    Returns:
    -------
        log(x)

    """
    return math.log(x)


# - exp
def exp(x: float) -> float:
    """Calculates the exponent of x

    Args:
    ----
        x: float number

    Returns:
    -------
        exp(x)

    """
    return math.exp(x)


# - log_back
def log_back(x: float, y: float) -> float:
    """Calculates the derivative of log at x multiplied by y

    Args:
    ----
        x: float number
        y: float number

    Returns:
    -------
        f'(x)

    """
    return 1 / x * y


# - inv
def inv(x: float) -> float:
    """Calculates the inverse of x
    Args:
        x: float number

    Returns
    -------
        1/x

    """
    if x == 0:
        raise ("Division by 0")
    return 1 / x


# - inv_back
def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg

    Args:
    ----
        x: Any number
        y: Any number

    Returns:
    -------
        f'(x) * y

    """
    return -y / (x**2)


# - relu_back
def relu_back(x: float, y: float) -> float:
    """Computes the derivative of relu times a second arg

    Args:
    ----
        x: Any number
        y: Any number

    Returns:
    -------
        f'(x) * y

    """
    if x <= 0:
        return 0
    return y


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(func: Callable[[float], float], iterable: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable"""
    return [func(element) for element in iterable]


# - zipWith
def zipWith(
    func: Callable[[float, float], float],
    iterable1: Iterable[float],
    iterable2: Iterable[float],
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function"""
    zipped = []
    assert len(iterable1) == len(iterable2)
    for i in range(len(iterable1)):
        zipped.append(func(iterable1[i], iterable2[i]))
    return zipped


# - reduce
def reduce(func: Callable[[float, float], float], iterable: Iterable[float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function

    Args:
    ----
        func: Given function
        iterable: Original list

    Returns:
    -------
        Reduced list

    """
    cml = iterable[0]
    for i in iterable[1:]:
        cml = func(cml, i)
    return cml


#
# Use these to implement
# - negList : negate a list
def negList(iterable: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list

    Args:
    ----
        iterable: Original list

    Returns:
    -------
        Negated list

    """
    return map(neg, iterable)


# - addLists : add two lists together
def addLists(iterable1: Iterable[float], iterable2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements of two lists

    Args:
    ----
        iterable1: list 1
        iterable2: list 2

    Returns:
    -------
        Added list

    """
    return zipWith(add, iterable1, iterable2)


# - sum: sum lists
def sum(iterable: Iterable[float]) -> float:
    """Sums the elemnts of a list

    Args:
    ----
        iterable: Original list

    Returns:
    -------
        Sum result

    """
    if len(iterable) == 0:
        return 0
    return reduce(add, iterable)


# - prod: take the product of lists
def prod(iterable: Iterable[float]) -> float:
    """Calculates the product of all elements in a list

    Args:
    ----
        iterable: Original list

    Returns:
    -------
        Product result

    """
    if len(iterable) == 0:
        return 0
    return reduce(mul, iterable)


# TODO: Implement for Task 0.3.
