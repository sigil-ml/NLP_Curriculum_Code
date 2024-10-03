"""Various optimizations algorithms for Machine Learning."""

import random as py_random
from typing import Callable

import jax.numpy as jnp
import jax.random as random
from jax import grad


def print_optimizer_results(theta1, theta2, loss):
    """Print the results of the optimization."""
    print(f"theta_pred = ({theta1:.3f}, {theta2:.3f}), loss = {loss:.3f}")


def sgd(
    f_pred: Callable,
    lr: float,
    max_n_iterations: int,
    loss_fn: Callable,
    examples: list[tuple[float, float, float]],
    convergence_criteria: float,
    key,
    return_history: bool = False,
) -> dict | tuple[float, float]:
    r"""Perform Stochastic Gradient Descent.

    Args:
        f_pred: The model function.
        lr: The learning rate.
        max_n_iterations: The maximum number of iterations.
        loss_fn: The loss function.
        examples: The examples.
        convergence_criteria: The convergence criteria.
        key: The random key.
        return_history: Whether to return the history.

    Returns:
        theta1s: The theta1 values recorded during optimization.
        theta2s: The theta2 values recorded during optimization.
        losses: The losses recorded during optimization.
        max_n_iterations: The maximum number of iterations

        or

        theta1: The final theta1 value.
        theta2: The final theta2 value
    """
    print(
        f"Running SGD with learning rate: {lr} and max iterations: {max_n_iterations}"
    )

    loss_theta1 = grad(loss_fn, argnums=0)
    loss_theta2 = grad(loss_fn, argnums=1)
    theta1, theta2 = random.uniform(key), random.uniform(key)
    prev_loss = 0
    theta1s, theta2s, losses = [], [], []
    for i in range(max_n_iterations):
        key, subkey = random.split(key)
        idx = random.randint(subkey, (1,), 0, len(examples))
        x, y, z = examples[idx[0]]
        theta1 -= lr * loss_theta1(theta1, theta2, f_pred, x, y, z)
        theta2 -= lr * loss_theta2(theta1, theta2, f_pred, x, y, z)
        loss = loss_fn(theta1, theta2, f_pred, x, y, z).item()
        theta1s.append(theta1.item())
        theta2s.append(theta2.item())
        losses.append(loss)
        if jnp.abs(loss - prev_loss) < convergence_criteria:
            print(f"Converged at iteration {i + 1}!")
            print_optimizer_results(theta1, theta2, loss)
            max_n_iterations = i
            break
        prev_loss = loss
        print_optimizer_results(theta1, theta2, loss)

    if return_history:
        return {
            "theta1s": theta1s,
            "theta2s": theta2s,
            "losses": losses,
            "max_n_iterations": max_n_iterations,
        }
    return theta1s[-1], theta2s[-1]


def sgd_batched(
    f_pred: Callable,
    lr: float,
    batch_size: int,
    max_n_iterations: int,
    loss_fn: Callable,
    examples: list[tuple[float, float, float]],
    convergence_criteria: float,
    key,
    return_history: bool = False,
) -> dict | tuple[float, float]:
    r"""Perform Stochastic Gradient Descent until convergence or `max_n_iterations` is reached.

    Args:
        f_pred: The model function.
        lr: The learning rate.
        batch_size: The batch size.
        max_n_iterations: The maximum number of iterations.
        loss_fn: The loss function.
        examples: The examples.
        convergence_criteria: The convergence criteria.
        key: The random key.
        return_history: Whether to return the history.

    Returns:
        theta1s: The theta1 values recorded during optimization.
        theta2s: The theta2 values recorded during optimization.
        losses: The losses recorded during optimization.
        max_n_iterations: The maximum number of iterations

        or

        theta1: The final theta1 value.
        theta2: The final theta2 value
    """
    print(
        f"Running SGD with learning rate: {lr} and max iterations: {max_n_iterations}"
    )

    loss_theta1 = grad(loss_fn, argnums=0)
    loss_theta2 = grad(loss_fn, argnums=1)
    theta1, theta2 = random.uniform(key), random.uniform(key)
    prev_loss = 0
    theta1s, theta2s, losses = [], [], []
    for i in range(max_n_iterations):
        theta1_sum_loss, theta2_sum_loss = 0, 0
        for _ in range(batch_size):
            x, y, z = py_random.choices(examples)[0]
            theta1_sum_loss += loss_theta1(theta1, theta2, f_pred, x, y, z)
            theta2_sum_loss += loss_theta2(theta1, theta2, f_pred, x, y, z)

        theta1_loss = theta1_sum_loss / batch_size
        theta2_loss = theta2_sum_loss / batch_size

        theta1 -= lr * theta1_loss
        theta2 -= lr * theta2_loss
        loss = loss_fn(theta1, theta2, f_pred, x, y, z).item()
        theta1s.append(theta1.item())
        theta2s.append(theta2.item())
        losses.append(loss)
        if jnp.abs(loss - prev_loss) < convergence_criteria:
            print(f"Converged at iteration {i + 1}!")
            print_optimizer_results(theta1, theta2, loss)
            max_n_iterations = i
            break
        prev_loss = loss
        print_optimizer_results(theta1, theta2, loss)

    if return_history:
        return {
            "theta1s": theta1s,
            "theta2s": theta2s,
            "losses": losses,
            "max_n_iterations": max_n_iterations,
        }
    return theta1s[-1], theta2s[-1]


def RMSProp(
    f_pred: Callable,
    lr: float,
    weight_decay: float,
    smoothing_constant: float,
    momentum: float,
    centered: bool,
    max_n_iterations: int,
    loss_fn: Callable,
    examples: list[tuple[float, float, float]],
    convergence_criteria: float,
    key,
    return_history: bool = False,
) -> dict | tuple[float, float]:
    r"""Perform RMSProp until convergence or `max_n_iterations` is reached.

    Algorithm taken from:
    https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

    Args:
        f_pred: The model function.
        lr: The learning rate.
        weight_decay: L2 penalty.
        smoothing_constant: The smoothing constant.
        momentum: The momentum factor.
        centered: Whether to use centered RMSProp.
        max_n_iterations: The maximum number of iterations.
        loss_fn: The loss function.
        examples: The examples.
        convergence_criteria: The convergence criteria.
        key: The random key.
        return_history: Whether to return the history.

    Returns:
        theta1s: The theta1 values recorded during optimization.
        theta2s: The theta2 values recorded during optimization.
        losses: The losses recorded during optimization.
        max_n_iterations: The maximum number of iterations

        or

        theta1: The final theta1 value.
        theta2: The final theta2 value
    """
    print(
        f"Running SGD with learning rate: {lr} and max iterations: {max_n_iterations}"
    )

    loss_theta1 = grad(loss_fn, argnums=0)
    loss_theta2 = grad(loss_fn, argnums=1)
    theta1, theta2 = random.uniform(key), random.uniform(key)
    prev_loss = 0
    theta1s, theta2s, losses = [], [], []
    square_avg_1, square_avg_2 = 0, 0
    buffer_1, buffer_2 = 0, 0
    g_avg_1, g_avg_2 = 0, 0
    for i in range(max_n_iterations):
        x, y, z = py_random.choices(examples)[0]
        g_t_1 = loss_theta1(theta1, theta2, f_pred, x, y, z)
        g_t_2 = loss_theta2(theta1, theta2, f_pred, x, y, z)

        if weight_decay != 0:
            g_t_1 += weight_decay * theta1
            g_t_2 += weight_decay * theta2

        square_avg_1 = smoothing_constant * square_avg_1 + (1 - lr) * g_t_1**2
        square_avg_2 = smoothing_constant * square_avg_2 + (1 - lr) * g_t_2**2

        square_avg_1_backup = square_avg_1
        square_avg_2_backup = square_avg_2

        if centered:
            g_avg_1 = smoothing_constant * g_avg_1 + (1 - lr) * g_t_1
            g_avg_2 = smoothing_constant * g_avg_2 + (1 - lr) * g_t_2
            square_avg_1_backup -= g_avg_1**2
            square_avg_2_backup -= g_avg_2**2

        if momentum > 0:
            buffer_1 = momentum * buffer_1 + g_t_1 / (
                jnp.sqrt(square_avg_1_backup) + 1e-8
            )
            buffer_2 = momentum * buffer_2 + g_t_2 / (
                jnp.sqrt(square_avg_2_backup) + 1e-8
            )
            theta1 -= lr * buffer_1
            theta2 -= lr * buffer_2
        else:
            theta1 -= lr * g_t_1 / (jnp.sqrt(square_avg_1) + 1e-8)
            theta2 -= lr * g_t_2 / (jnp.sqrt(square_avg_2) + 1e-8)

        loss = loss_fn(theta1, theta2, f_pred, x, y, z).item()
        theta1s.append(theta1.item())
        theta2s.append(theta2.item())
        losses.append(loss)
        if jnp.abs(loss - prev_loss) < convergence_criteria:
            print(f"Converged at iteration {i + 1}!")
            print_optimizer_results(theta1, theta2, loss)
            max_n_iterations = i
            break
        prev_loss = loss
        print_optimizer_results(theta1, theta2, loss)

    if return_history:
        return {
            "theta1s": theta1s,
            "theta2s": theta2s,
            "losses": losses,
            "max_n_iterations": max_n_iterations,
        }
    else:
        return theta1s[-1], theta2s[-1]


def Adam(
    f_pred: Callable,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    max_n_iterations: int,
    loss_fn: Callable,
    examples: list[tuple[float, float, float]],
    convergence_criteria: float,
    key,
    return_history: bool = False,
) -> dict | tuple[float, float]:
    r"""Perform Adam Optimization until convergence criterion is reached or `max_n_iterations`.

    Algorithm taken from:
    https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

    Args:
        f_pred: The model function.
        lr: The learning rate.
        weight_decay: L2 penalty.
        betas: First and second moment exponential decay rates.
        max_n_iterations: The maximum number of iterations.
        loss_fn: The loss function.
        examples: The examples.
        convergence_criteria: The convergence criteria.
        key: The random key.
        return_history: Whether to return the history.

    Returns:
        theta1s: The theta1 values recorded during optimization.
        theta2s: The theta2 values recorded during optimization.
        losses: The losses recorded during optimization.
        max_n_iterations: The maximum number of iterations

        or

        theta1: The final theta1 value.
        theta2: The final theta2 value
    """
    thetas = [random.uniform(key), random.uniform(key)]
    prev_loss = 0
    theta1s, theta2s, losses = [], [], []
    beta1, beta2 = betas
    n_params = 2
    first_moments = [0, 0]
    second_moments = [0, 0]
    bias_corrected_first_moments = [0, 0]
    bias_corrected_second_moments = [0, 0]
    for i in range(1, max_n_iterations):
        x, y, z = py_random.choices(examples)[0]

        for param_idx in range(n_params):
            gradient = grad(loss_fn, argnums=param_idx)(
                thetas[0], thetas[1], f_pred, x, y, z
            )

            theta = thetas[param_idx]
            fm = first_moments[param_idx]
            sm = second_moments[param_idx]
            bc_fm = bias_corrected_first_moments[param_idx]
            bc_sm = bias_corrected_second_moments[param_idx]

            # Update the parameters

            if weight_decay != 0:
                gradient += weight_decay * theta

            fm = beta1 * fm + (1 - beta1) * gradient
            sm = beta2 * sm + (1 - beta2) * gradient**2

            # bias corrections
            bc_fm = fm / (1 - beta1**i)
            bc_sm = sm / (1 - beta2**i)

            # 1e-8 is added to avoid division by zero
            theta -= lr * bc_fm / (jnp.sqrt(bc_sm) + 1e-8)

            thetas[param_idx] = theta
            first_moments[param_idx] = fm
            second_moments[param_idx] = sm
            bias_corrected_first_moments[param_idx] = bc_fm
            bias_corrected_second_moments[param_idx] = bc_sm

        loss = loss_fn(thetas[0], thetas[1], f_pred, x, y, z).item()
        theta1s.append(thetas[0].item())
        theta2s.append(thetas[1].item())
        losses.append(loss)
        if jnp.abs(loss - prev_loss) < convergence_criteria:
            print(f"Converged at iteration {i + 1}!")
            print_optimizer_results(thetas[0], thetas[1], loss)
            max_n_iterations = i
            break
        prev_loss = loss
        print_optimizer_results(thetas[0], thetas[1], loss)

    if return_history:
        return {
            "theta1s": theta1s,
            "theta2s": theta2s,
            "losses": losses,
            "max_n_iterations": max_n_iterations,
        }
    return theta1s[-1], theta2s[-1]
