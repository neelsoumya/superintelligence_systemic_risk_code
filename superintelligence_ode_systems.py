import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
alpha = {'H': 0.1, 'A': 0.1, 'G': 0.1, 'I': 0.1}
beta = {
    ('H','A'): 0.05, ('H','G'): 0.05, ('H','I'): 0.05,
    ('A','H'): 0.05, ('A','G'): 0.05, ('A','I'): 0.05,
    ('G','H'): 0.05, ('G','A'): 0.05, ('G','I'): 0.05,
    ('I','H'): 0.05, ('I','A'): 0.05, ('I','G'): 0.05
}
gamma = {'H': 0.1, 'A': 0.1, 'G': 0.1, 'I': 0.1}
delta = {'H': 0.02, 'A': 0.02, 'G': 0.02, 'I': 0.02}
theta = {'H': 0.01, 'A': 0.01, 'G': 0.01, 'I': 0.01}
rho = 0.5
kappa = 0.1

def system(t, y):
    H, A, G, I, S = y
    dH = -alpha['H']*H + beta[('H','A')]*A + beta[('H','G')]*G + beta[('H','I')]*I + gamma['H']*S - delta['H']*S*H
    dA = -alpha['A']*A + beta[('A','H')]*H + beta[('A','G')]*G + beta[('A','I')]*I + gamma['A']*S - delta['A']*S*A
    dG = -alpha['G']*G + beta[('G','H')]*H + beta[('G','A')]*A + beta[('G','I')]*I + gamma['G']*S - delta['G']*S*G
    dI = -alpha['I']*I + beta[('I','H')]*H + beta[('I','A')]*A + beta[('I','G')]*G + gamma['I']*S - delta['I']*S*I
    dS = rho - kappa*S - (theta['H']*H + theta['A']*A + theta['G']*G + theta['I']*I)*S
    return [dH, dA, dG, dI, dS]

# Initial conditions: H, A, G, I, S
y0 = [1.0, 1.0, 1.0, 1.0, 0.5]
t_span = (0, 100)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], label='Healthcare (H)')
plt.plot(sol.t, sol.y[1], label='Agriculture (A)')
plt.plot(sol.t, sol.y[2], label='Governance (G)')
plt.plot(sol.t, sol.y[3], label='Infrastructure (I)')
plt.plot(sol.t, sol.y[4], label='AI System (S)', linestyle='--', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Level')
plt.title('Dynamic AI-Driven System in Developing Nations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('figures/ai_driven_system.png', dpi=300)
# close the plot to avoid display in Jupyter Notebook
plt.close()
