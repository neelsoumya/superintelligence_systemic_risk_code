import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
alpha = 1.0   # Base learning rate
beta = 2.0    # Self-improvement exponent (> 1)
K = 100.0     # Cognitive saturation threshold

# Differential equation: dI/dt = (alpha * I^beta) / (1 + I / K)
def dI_dt(t, I):
    return (alpha * I**beta) / (1 + I / K)

# Time span for the simulation
t_span = (0, 10)  # simulate from t=0 to t=10
t_eval = np.linspace(*t_span, 1000)  # times at which to store the computed results

# Initial condition
I0 = [0.1]  # starting intelligence level

# Solve the differential equation
sol = solve_ivp(dI_dt, t_span, I0, t_eval=t_eval, method='RK45')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Intelligence level $I(t)$')
plt.axhline(K, color='gray', linestyle='--', label='Saturation threshold $K$')
plt.title("Recursive Self-Improvement of AGI")
plt.xlabel("Time $t$")
plt.ylabel("Intelligence $I(t)$")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('figures/3phase.png', dpi=300)
plt.close()