# version of 27 November

import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def optimize_path(
    h: Callable[[jnp.ndarray], jnp.ndarray],
    alpha: Callable[[jnp.ndarray], jnp.ndarray],
    beta:float,
    weight_jump:float,
    max_steepness:float,
    gamma_start: Tuple[float, float],
    gamma_end: Tuple[float, float],
    N: int = 128,
    maxiter: int = 100,
    verbose: int = 1,
):
    """
    h:              altitude function
    alpha:          friction function
    beta:           adimensional parameter for ration [alpha]/[h]
    weight_jump     penalisation on far points of the path
    gamma_start     starting point
    gamma_end       final point
    N:              discretisation of gamma
    maxiter:        max iteration of minimize
    """
    assert N >= 2
    
    # Initial points
    x0 = jnp.array(gamma_start, dtype=jnp.float32)
    x1 = jnp.array(gamma_end, dtype=jnp.float32)
    dt = 1.0 / (N - 1)

    # number of interior points 
    interior_count = (N - 2) * 2  # number of scalar variables
    
    # initial segment 
    initial_segment = jnp.linspace(x0, x1, N)

    # Small random noise on the initial segment,
    # This may help if the segment is already a local max/min
    noise_scale = .0 # turn this on to 0.01 to perturbe the initial segment
    init_interior = np.array(initial_segment[1:-1].reshape(-1))
    init_interior += np.random.normal(0, noise_scale, size=init_interior.shape)

    # flattened interior numpy array (but we'll convert to jnp inside)
    def assemble_full_path(flat_interior):
        """flat_interior: shape ((N-2)*2,)"""
        interior = flat_interior.reshape((N - 2, 2))
        full = jnp.vstack([x0, interior, x1])  # shape (N,2)
        return full

    def jax_objective(flat_interior):

        def exploding_threshold(x, threshold=1.5, steepness=10.0):
            excess = jax.nn.relu(x - threshold)
            return jnp.exp(steepness * excess) - 1.0
        # Reconstruct the full curve
        p = assemble_full_path(flat_interior)        # shape (N,2)

        # --- geometric quantities ---
        diffs       = p[1:] - p[:-1]                 # γ' approx, shape (N-1,2)
        seg_lengths2= jnp.sum(diffs**2, axis=1)
        seg_lengths = jnp.sqrt(seg_lengths2 + 1e-12)

        # midpoints (for evaluation)
        midpoints = 0.5 * (p[:-1] + p[1:])           # shape (N-1,2)
        
        # alpha on midpoints
        vmap_alpha = jax.vmap(alpha)
        mid_alpha = vmap_alpha(midpoints)

        # grad_h on midpoints
        vmap_gradh = jax.vmap(jax.grad(h))
        midgradh = vmap_gradh(midpoints)

        # total lenght
        sum_lenght=jnp.sum(seg_lengths)
        avg_lenght=sum_lenght/N

        # Integral energies
        #-------------------

        friction = jnp.sum(mid_alpha * seg_lengths)
        
        scalarpdt = jnp.einsum('ij,ij->i', midgradh, diffs)
        #elevation = beta*jnp.sum(jax.nn.relu(scalarpdt))
        elevation = beta*jnp.sum(jnp.abs(scalarpdt))

        # Calculate the variance of the segment lengths
        mean_len = jnp.mean(seg_lengths)
        # The penalty is the sum of squared differences from the mean
        jump_penalization = jnp.sum((seg_lengths - mean_len)**2) / N

        # New regularisation
        length_regularization = 0.1/N**2 * jnp.sum( 1./ seg_lengths) #+ jnp.sum(seg_lengths)

        # Total energy
        total = friction + elevation + weight_jump*jump_penalization # + length_regularization #+ 100*steepness_penalization # 

        return total


    # JIT and value_and_grad for speed
    jax_obj_and_grad = jax.jit(jax.value_and_grad(jax_objective))

    # SciPy wrapper: scipy passes numpy arrays; we convert to jnp inside and return numpy scalars/grad
    def scipy_objective(x_numpy):
        x_jnp = jnp.array(x_numpy, dtype=jnp.float32)
        val, grad = jax_obj_and_grad(x_jnp)
        return np.float32(val), np.array(grad, dtype=float)

    # Bounds: keep interior points inside [0,1]^2
    bounds = [(0.0, 1.0)] * interior_count

    # Run SciPy minimize (L-BFGS-B)
    res = minimize(
        fun=lambda x: scipy_objective(x)[0],
        x0=init_interior,
        jac=lambda x: scipy_objective(x)[1],
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': maxiter,
            'disp': bool(verbose),
            'ftol': 1e-14,   # Relaxed tolerance
            'gtol': 1e-12,   # Gradient tolerance
            'maxls': 50     # More line-search attempts
        },
    )

    final_interior = res.x.reshape((N - 2, 2))
    full_path = np.vstack([np.array(x0), final_interior, np.array(x1)])
    out = {
        'gamma': full_path,
        'fun': res.fun,
        'success': res.success,
        'message': res.message,
        'nit': res.nit,
    }
    return out

def plot_path(h, result):
    # Plotting 

    # --- gradient of h ---
    grad_h = jax.jit(jax.grad(h))       
    vmap_grad_h = jax.vmap(grad_h)

    # Compute meshgrid
    nx, ny = 100, 100
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    points = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # Evaluate h
    Z_flat = jax.vmap(h)(jnp.array(points))
    Z = Z_flat.reshape((ny, nx))

    # Evaluate grad h
    grad_flat = vmap_grad_h(jnp.array(points))  
    G = jnp.sqrt(jnp.sum(grad_flat**2, axis=1))
    G = G.reshape((ny, nx))

    # Path
    gamma = np.array(result['gamma'])
    gamma_z = jax.vmap(h)(jnp.array(gamma))

    # --------------------------
    # PLOTS
    # --------------------------

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "xy"}]],
    )

    # --- 3D surface ---
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.9,
        ),
        row=1, col=1
    )

    # --- Path on 3D surface ---
    fig.add_trace(
        go.Scatter3d(
            x=gamma[:,0],
            y=gamma[:,1],
            z=gamma_z,
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=3, color='red'),
            name=''
        ),
        row=1, col=1
    )
    # --- Contour plot ---
    fig.add_trace(
        go.Contour(
            x=x, y=y, z=Z,
            colorscale='Viridis',
            contours=dict(showlines=False),
        ),
        row=1, col=2
    )

    # --- Path on contour plot ---
    fig.add_trace(
        go.Scatter(
            x=gamma[:,0],
            y=gamma[:,1],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=4, color='red'),
            name=''
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=600,
        width=1200,
    )

    fig.show()    



if __name__ == "__main__":

    def h(xy, sigma=0.15,sigma1=0.5):
        x, y = xy[0], xy[1]

        xc0, yc0 = 0.1, 0.1
        xc1, yc1 = 0.9, 0.9
        xc2, yc2 = 1, 0

        r0 = (x - xc0)**2 + (y - yc0)**2
        r1 = (x - xc1)**2 + (y - yc1)**2
        r2 = (x - xc2)**2 + (y - yc2)**2
        
        # Incomment below for different landscapes:

        #return 1+0*jnp.exp(-r0/sigma**2)-jnp.exp(-r1/sigma1**2)+jnp.sin(2*3.14*x) * jnp.sin(2*3.14*y)
        return 1+ jnp.exp( -5*r1)*jnp.sin(3*x)**2-jnp.exp( -1*r2)  # WORKS VERY WELL!!!
        #return 1+ (0.7*jnp.exp(-5*r0 )+0.7*jnp.exp(-5*r1)+0.7*jnp.exp(-5*r2))+.2*(jnp.sin(4*x)**2*jnp.sin(4*y)**2)

    def alpha(xy,):
      x, y = xy[0], xy[1]
      return 0.5


    start = (0.1, 0.4)
    end = (0.9, 0.9)

    beta=3.0
    weight_jump=1000
    max_steepness=1.5
    result = optimize_path(h, alpha, beta,weight_jump,max_steepness,start, end, N=80, maxiter=10000, verbose=10)

    # Save the path in a file
    with open("path.pkl", 'wb') as file:
        pickle.dump(result['gamma'], file)

    print("Optimization success:", result['success'], "nit:", result['nit'])
    print("Final objective:", result['fun'])

    plot_path(h, result)

