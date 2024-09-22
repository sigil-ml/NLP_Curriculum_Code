import jax.numpy as jnp
import jax.random as random
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from typing import Callable

pio.templates.default = "plotly_white"


def generate_examples(fn, n_examples, key) -> list[tuple]:
    r"""Generates `n_examples` examples from the function `fn`."""
    examples = []
    for i in range(n_examples):
        key, subkey = random.split(key)
        x = random.uniform(subkey, (1,), minval=-10, maxval=10)
        key, subkey = random.split(key)
        y = random.uniform(subkey, (1,), minval=-10, maxval=10)
        z = fn(x, y)
        examples.append((x, y, z))
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
            [{"type": "surface"}, {"type": "surface"}],
        ],
    )

    """
    Create a 2D surface plot of the true function
    """
    # TODO: Do not have hard coded values
    xs = jnp.arange(-10, 10, 0.1)
    ys = jnp.arange(-10, 10, 0.1)
    zs = jnp.array([fn(x, ys) for x in xs])
    z_min, z_max = jnp.min(zs), jnp.max(zs)
    z_range = z_max - z_min
    z_axes_min = z_min - 0.1 * z_range
    z_axes_max = z_max + 0.1 * z_range

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[-12, 12],
            ),
            yaxis=dict(
                range=[-12, 12],
            ),
            zaxis=dict(
                range=[z_axes_min, z_axes_max],
            ),
        )
    )

    true_function_surface = go.Surface(
        z=zs, x=xs, y=ys, colorscale="Blues", showscale=False
    )
    fig.add_trace(true_function_surface, row=1, col=1)

    ######################################################################################
    # Animated y_pred surface plot
    ######################################################################################
    steps = []
    for i, f_pred in enumerate(f_preds):
        pred_zs = [f_pred(x, ys) for x in xs]
        fig.add_trace(
            go.Surface(
                z=pred_zs,
                x=xs,
                y=ys,
                visible=False,
                name=f"Iterations {(i + 1):4d}",
                showscale=False,
            ),
            row=1,
            col=1,
        )
        step = dict(
            method="update",
            args=[
                {"visible": [False] * (n_iterations + 1)},
                {"title": "SGD Iteration: " + str(i + 1)},
            ],
        )
        step["args"][0]["visible"][0] = True
        step["args"][0]["visible"][i] = True
        steps.append(step)

    """
    Create the loss surface plot with the path of the optimizer and the convergence circle
    """
    ######################################################################################
    # Adds the convergence circle to the loss surface plot
    ######################################################################################
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
    fig.add_trace(circle_trace, row=1, col=2)
    ######################################################################################
    # Adds the loss surface to the plot
    ######################################################################################
    theta_1_max = jnp.max(jnp.array(theta1s))
    theta_2_max = jnp.max(jnp.array(theta2s))

    theta_1_min = jnp.min(jnp.array(theta1s))
    theta_2_min = jnp.min(jnp.array(theta2s))

    if theta_1_max > theta_2_max:
        theta_max = theta_1_max
    else:
        theta_max = theta_2_max

    if theta_1_min < theta_2_min:
        theta_min = theta_1_min
    else:
        theta_min = theta_2_min

    gtheta_range = jnp.absolute(theta_max - theta_min)
    gtheta_step = gtheta_range / 100

    gtheta1s = jnp.arange(
        theta_min - 0.2 * gtheta_range, theta_max + 0.2 * gtheta_range, gtheta_step
    )

    gtheta2s = jnp.arange(
        theta_min - 0.2 * gtheta_range, theta_max + 0.2 * gtheta_range, gtheta_step
    )

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

    z_loss_arr = jnp.array(z_loss)
    z_loss_min, z_loss_max = jnp.min(z_loss_arr), jnp.max(z_loss_arr)
    z_loss_range = jnp.absolute(z_loss_max - z_loss_min)
    z_loss_arr -= 0.01 * z_loss_range
    z_loss = z_loss_arr.tolist()

    fig.add_trace(
        go.Surface(
            x=gtheta1s,
            y=gtheta2s,
            z=gaussian_filter(z_loss, sigma=[2, 2]),
            showscale=False,
        ),
        row=1,
        col=2,
    )
    ######################################################################################
    # Adds the optimizer path to the plot
    ######################################################################################
    fig.add_trace(
        go.Scatter3d(
            x=theta1s,
            y=theta2s,
            z=losses,
            line=dict(color="lightblue", width=6),
            marker=dict(
                size=3,
                color=losses,
                colorscale="Blues",
            ),
        ),
        row=1,
        col=2,
    )
    ######################################################################################
    """
    Adds animations to the plot
    """
    ######################################################################################
    sliders = [
        dict(
            active=n_iterations,
            currentvalue={"prefix": "SGD: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.update_layout(
        title="SGD 3D",
        autosize=True,
        sliders=sliders,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
    )
    ######################################################################################
    return fig
