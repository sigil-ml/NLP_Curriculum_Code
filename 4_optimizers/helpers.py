import jax.numpy as jnp
from jax import jit
import jax.random as random
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from typing import Callable
from contextlib import contextmanager
import time
import numpy as np
import shutil
import inspect


pio.templates.default = "plotly_white"


@contextmanager
def timer(code_block: str, perf_profiling: bool = True):
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
                {"type": "surface"},
            ],  # this is needed or the make_subplots will fail
        ],
    )

    # These two lists will hold the traces for our animation logic. The approximated list will start
    # with the true function and then add the approximated function for each iteration. The optimization
    # will start with the loss surface and then add the optimizer path for each iteration.
    approximated_fn_traces = []
    optimization_path_traces = []
    slider_steps = []

    """
    Create a 2D surface plot of the true function
    """
    # TODO: Do not have hard coded values
    with timer("True 3D Plot Creation", perf_profiling):
        xs = jnp.arange(-10, 10, 0.1)
        ys = jnp.arange(-10, 10, 0.1)
        zs = jnp.array([fn(x, ys) for x in xs])
        z_min, z_max = jnp.min(zs), jnp.max(zs)
        z_range = z_max - z_min
        z_axes_min = z_min - 0.1 * z_range
        z_axes_max = z_max + 0.1 * z_range

        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(
        #             range=[-12, 12],
        #         ),
        #         yaxis=dict(
        #             range=[-12, 12],
        #         ),
        #         zaxis=dict(
        #             range=[z_axes_min, z_axes_max],
        #         ),
        #     )
        # )

        true_function_surface = go.Surface(
            z=zs, x=xs, y=ys, colorscale="Blues", showscale=False
        )
        approximated_fn_traces.append(true_function_surface)
        # fig.add_trace(true_function_surface, row=1, col=1)

    ######################################################################################
    # Animated y_pred surface plot
    ######################################################################################
    with timer("Predicted 3D Animation Creation", perf_profiling):
        # visibilities = [False] * (n_iterations + 1)
        for i, f_pred in enumerate(f_preds):
            print(inspect.getsource(f_pred))
            pred_zs = [f_pred(x, ys) for x in xs]
            approximated_surface_trace = go.Surface(
                z=pred_zs,
                x=xs,
                y=ys,
                # visible=False,
                # name=f"Iterations {(i + 1):4d}",
                showscale=False,
            )
            approximated_fn_traces.append(approximated_surface_trace)
            # fig.add_trace(estimated_surface_trace, row=1, col=1)
            # step = dict(
            #     method="update",
            #     args=[
            #         {"visible": visibilities},
            #         {"title": "SGD Iteration: " + str(i + 1)},
            #     ],
            # )
            # step["args"][0]["visible"][0] = True
            # step["args"][0]["visible"][i] = True
            # animation_steps.append(step)

    """
    Create the loss surface plot with the path of the optimizer and the convergence circle
    """
    ######################################################################################
    # Adds the convergence circle to the loss surface plot
    ######################################################################################
    with timer("Convergence Circle Plot Creation", perf_profiling):
        _theta = jnp.linspace(0, 2 * jnp.pi, 100)
        r = convergence_criteria * 50
        x_circle = true_thetas[0] + r * jnp.cos(_theta)
        y_circle = true_thetas[1] + r * jnp.sin(_theta)

        z_circle = jnp.sin(x_circle) * jnp.cos(y_circle)
        z_circle *= 0.2 * jnp.max(z_circle)

        circle_trace = go.Scatter3d(
            x=x_circle,
            y=y_circle,
            z=z_circle,
            mode="lines",
            line=dict(color="lightgreen", width=10),
            name="Convergence Radius",
        )
        optimization_path_traces.append(circle_trace)
    ######################################################################################
    # Adds the loss surface to the plot
    ######################################################################################
    with timer("Loss Surface Plot Creation", perf_profiling):
        theta1s = jnp.array(theta1s)
        theta2s = jnp.array(theta2s)
        thetas = jnp.concatenate([theta1s, theta2s])

        # theta_1_max = jnp.max(jnp.array(theta1s))
        # theta_2_max = jnp.max(jnp.array(theta2s))

        # theta_1_min = jnp.min(jnp.array(theta1s))
        # theta_2_min = jnp.min(jnp.array(theta2s))

        # if theta_1_max > theta_2_max:
        #     theta_max = theta_1_max
        # else:
        #     theta_max = theta_2_max

        # if theta_1_min < theta_2_min:
        #     theta_min = theta_1_min
        # else:
        #     theta_min = theta_2_min

        theta_min, theta_max = thetas.min(), thetas.max()
        gtheta_range = jnp.absolute(theta_max - theta_min)
        # gtheta_step = gtheta_range / 10

        gtheta1s = jnp.arange(
            theta_min - 0.2 * gtheta_range,
            theta_max + 0.2 * gtheta_range,
            gtheta_range / 10,
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
        # z_loss_min, z_loss_max = jnp.min(z_loss_arr), jnp.max(z_loss_arr)
        z_loss_range = jnp.absolute(z_loss.max() - z_loss.min())
        z_loss -= 0.01 * z_loss_range
        z_loss = z_loss.tolist()

        loss_surface_trace = go.Surface(
            x=gtheta1s,
            y=gtheta2s,
            z=gaussian_filter(z_loss, sigma=[1, 1]),
            showscale=False,
        )
        optimization_path_traces.append(loss_surface_trace)
        # fig.add_trace(loss_surface_trace, row=1, col=2)
    ######################################################################################
    # Adds the optimizer path to the plot
    ######################################################################################
    with timer("Optimizer Path Plot Creation", perf_profiling):
        for i in range(n_iterations):
            optimizer_scatter_trace = go.Scatter3d(
                x=theta1s[: i + 1],
                y=theta2s[: i + 1],
                z=losses[: i + 1],
                line=dict(color="lightblue", width=6),
                marker=dict(
                    size=3,
                    color=losses[: i + 1],
                    colorscale="Blues",
                ),
                name="Optimizer Path",
            )
            optimization_path_traces.append(optimizer_scatter_trace)
            # fig.add_trace(optimizer_scatter_trace, row=1, col=2)
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=theta1s,
        #         y=theta2s,
        #         z=losses,
        #         line=dict(color="lightblue", width=6),
        #         marker=dict(
        #             size=3,
        #             color=losses,
        #             colorscale="Blues",
        #         ),
        #     ),
        #     row=1,
        #     col=2,
        # )
    ######################################################################################
    """
    Adds animations to the plot
    """
    ######################################################################################
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
            step = dict(
                method="animate",
                args=[
                    [f"Optimization Step {i}"],
                    {
                        "frame": {"duration": 200, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                label=f"Optimization Step {i}",
            )
            slider_steps.append(step)

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "Step: "},
                pad={"t": 50},
                steps=slider_steps,
            )
        ]

        updatemenus = [
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
                direction="left",
                pad={"r": 10, "t": 70},
                showactive=False,
            )
        ]

        fig.update_layout(
            sliders=sliders,
            updatemenus=updatemenus,
        )

        # sliders = [
        #     dict(
        #         active=n_iterations,
        #         currentvalue={"prefix": "SGD: "},
        #         pad={"t": 50},
        #         steps=animation_steps,
        #     )
        # ]

        # fig.data[0].visible = True
        # fig.data[1].visible = True
        # fig.update_layout(
        #     title="SGD 3D",
        #     autosize=True,
        #     sliders=sliders,
        #     height=800,
        #     margin=dict(l=65, r=50, b=65, t=90),
        # )
    ######################################################################################
    return fig
