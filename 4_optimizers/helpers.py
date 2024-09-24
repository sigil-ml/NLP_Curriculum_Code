import inspect
import shutil
import time
from contextlib import contextmanager
from typing import Callable

import jax.numpy as jnp
import jax.random as random
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# from jax import jit
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

pio.templates.default = "plotly_white"


@contextmanager
def timer(code_block: str, perf_profiling: bool = True):
    r"""Logs the time elapsed for a code block to execute."""
    term_cols, _ = shutil.get_terminal_size()
    if perf_profiling:
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        n_dots = term_cols - len(code_block)
        print(f"[{code_block}] {n_dots * '.'} Elapsed time: {(end - start):.3f} (s)")
    else:
        yield


def generate_examples(
    fn, n_examples, key, xbounds: tuple = (-10, 10), ybounds: tuple = (-10, 10)
) -> list[tuple]:
    r"""Generates `n_examples` examples from the function `fn`."""

    key, subkey1, subkey2 = random.split(key, num=3)
    xmin, xmax = xbounds
    ymin, ymax = ybounds
    n_examples = jnp.sqrt(n_examples).astype(int)
    xs = random.uniform(subkey1, (n_examples,), minval=xmin, maxval=xmax)
    ys = random.uniform(subkey2, (n_examples,), minval=ymin, maxval=ymax)
    zs = fn(xs[:, None], ys[None, :])
    x_grid, y_grid = jnp.meshgrid(xs, ys, indexing="ij")
    x_flat = np.array(x_grid).ravel()
    y_flat = np.array(y_grid).ravel()
    z_flat = np.array(zs).ravel()
    examples = list(zip(x_flat, y_flat, z_flat))

    return examples


def create_optimizer_figure_2d(
    fn: Callable,
    loss_fn: Callable,
    true_thetas: tuple[float | int, float | int],
    convergence_criteria: float,
    theta1s: list[float | int],
    theta2s: list[float | int],
    losses: list[float],
    f_preds: list[Callable],
    n_iterations: int,
    perf_profiling: bool = False,
) -> go.Figure:
    r"""This function creates a plotly figure that contains two columns of plots. The
    column is a 3D surface plot of the function along with the predicted function. The
    second column is a 3D surface plot of the loss function along with the path of the
    optimizer and a circle showing the convergence criteria. The plot has a slider for
    each iteration of the optimizer, updating the plot with the optimizer's path.
    """

    # If you do not include the specs the figure will fail to build
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [
                {"type": "surface"},
                {"type": "contour"},
            ],  # this is needed or the make_subplots will fail
        ],
    )
    fig.update_layout(height=600)

    # These two lists will hold the traces for our animation logic. The approximated list will start
    # with the true function and then add the approximated function for each iteration. The optimization
    # will start with the loss surface and then add the optimizer path for each iteration.
    approximated_fn_traces = []
    optimization_path_traces = []
    slider_steps = []

    # TODO: Do not have hard coded values
    with timer("True 3D Plot Creation", perf_profiling):
        xs = jnp.arange(-10, 10, 0.1)
        ys = jnp.arange(-10, 10, 0.1)
        zs = jnp.array([fn(x, ys) for x in xs])
        true_function_surface = go.Surface(
            z=zs, x=xs, y=ys, colorscale="Blues", showscale=False
        )
        approximated_fn_traces.append(true_function_surface)

    with timer("Predicted 3D Animation Creation", perf_profiling):
        for i, f_pred in enumerate(f_preds):
            theta1 = theta1s[i]
            theta2 = theta2s[i]
            pred_zs = [f_pred(x, ys, theta1, theta2) for x in xs]
            approximated_surface_trace = go.Surface(
                z=pred_zs, x=xs, y=ys, showscale=False, colorscale="OrRd"
            )
            approximated_fn_traces.append(approximated_surface_trace)

    with timer("Convergence Circle Plot Creation", perf_profiling):
        _theta = jnp.linspace(0, 2 * jnp.pi, 100)

        r = convergence_criteria * 50
        x_circle = true_thetas[0] + r * jnp.cos(_theta)
        y_circle = true_thetas[1] + r * jnp.sin(_theta)

        z_circle = jnp.sin(x_circle) * jnp.cos(y_circle)
        z_circle *= 0.2 * jnp.max(z_circle)

        circle_trace = go.Scatter(
            x=x_circle,
            y=y_circle,
            mode="lines",
            line={"color": "lightgreen", "width": 3},
            name="Convergence Radius",
            showlegend=False,
        )
        optimization_path_traces.append(circle_trace)

    with timer("Loss Surface Plot Creation", perf_profiling):
        theta1s = jnp.array(theta1s)
        theta2s = jnp.array(theta2s)
        thetas = jnp.concatenate([theta1s, theta2s])

        theta_min, theta_max = thetas.min(), thetas.max()
        gtheta_range = jnp.absolute(theta_max - theta_min)

        gtheta1s = jnp.arange(
            theta_min - 0.2 * gtheta_range,
            theta_max + 0.2 * gtheta_range,
            gtheta_range / 20,
        )

        gtheta2s = gtheta1s.copy()

        z_loss = []
        n_examples = len(gtheta1s) * len(gtheta2s)
        examples = generate_examples(fn, n_examples, random.PRNGKey(0))

        for theta1 in gtheta1s:
            row_loss = []
            for theta2 in gtheta2s:
                x, y, z = examples.pop()
                loss = loss_fn(theta2, theta1, x, y, z).item()
                row_loss.append(loss)
            z_loss.append(row_loss)

        z_loss = jnp.array(z_loss)
        z_loss_range = jnp.absolute(z_loss.max() - z_loss.min())
        z_loss -= 0.01 * z_loss_range
        z_loss = z_loss.tolist()

        loss_surface_trace = go.Contour(
            x=gtheta1s,
            y=gtheta2s,
            z=z_loss,
            showscale=False,
        )
        optimization_path_traces.append(loss_surface_trace)

    with timer("Optimizer Path Plot Creation", perf_profiling):
        for i in range(n_iterations):
            optimizer_scatter_trace = go.Scatter(
                x=theta1s[: i + 1],
                y=theta2s[: i + 1],
                line={"color": "orange", "width": 3},
                marker={
                    "size": 4,
                    "color": losses[: i + 1],
                    "colorscale": "OrRd",
                },
                name="Optimizer Path",
                showlegend=False,
            )
            optimization_path_traces.append(optimizer_scatter_trace)

    with timer("Animation Logic", perf_profiling):
        # Add initial traces to the figure
        fig.add_trace(approximated_fn_traces[0], row=1, col=1)  # True function
        fig.add_trace(optimization_path_traces[0], row=1, col=2)  # Loss surface
        fig.add_trace(optimization_path_traces[1], row=1, col=2)  # Convergence Circle

        fig.add_trace(approximated_fn_traces[1], row=1, col=1)
        fig.add_trace(optimization_path_traces[2], row=1, col=2)
        # Add approximated function/optimizer steps to the figure
        frames = []
        n_frames = len(optimization_path_traces)
        for i in range(1, n_frames):
            frame = go.Frame(
                data=[
                    approximated_fn_traces[i],
                    optimization_path_traces[i],
                ],
                traces=[3, 4],
                name=f"Optimization Step {i}",
            )
            frames.append(frame)

        fig.frames = frames

        # Create the slider steps
        for i in range(n_frames):
            step = {
                "method": "animate",
                "args": [
                    [f"Optimization Step {i}"],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": f"Optimization Step {i}",
            }
            slider_steps.append(step)

        sliders = [
            {
                "active": 0,
                "currentvalue": {"prefix": "Step: "},
                "pad": {"t": 50},
                "steps": slider_steps,
            }
        ]

        fig.update_layout(sliders=sliders)

    return fig
