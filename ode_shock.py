
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1000  # Time steps
dt = 0.1
time = np.linspace(0, T*dt, T)

#print(time)

# Sectors: Health, Agriculture, Governance
sectors = ["Health", "Agriculture", "Governance"]
n = len(sectors)

# Initial values
S = np.zeros((T, n))
S[0] = [0.6, 0.7, 0.5]  # initial health, agriculture, governance scores
A = np.zeros(T)
A[0] = 1.0  # initial AI capability

# Define parameters for each sector
alpha = np.array([0.05, 0.04, 0.03])  # decay
gamma = np.array([0.06, 0.08, 0.05])  # AI contribution
delta = np.array([0.02, 0.03, 0.025])  # AI-induced fragility
theta = np.array([0.01, 0.015, 0.02])  # sector feedback on AI

# Inter-sector influence matrix: beta[x, y] = influence of y on x
beta = np.array([
    [0.0, 0.04, 0.03],  # Health
    [0.02, 0.0, 0.01],  # Agriculture
    [0.01, 0.02, 0.0]   # Governance
])

# AI dynamics parameters
rho = 0.05  # investment rate
kappa = 0.02  # obsolescence rate

# Optional policy flags
policy_AI_boost = False
policy_shield_agriculture = False

# Simulate
for t in range(1, T):
    stress_factor = 1.0

    # Apply a sudden shock to AI at t=30
    if t == 30:
        A[t-1] *= 0.5  # drop AI capability suddenly
        print("Shock applied to AI at t=30")

    # Apply a governance collapse shock at t=60
    #if t == 60:
    #    S[t-1, 2] *= 0.3  # governance collapses
    #    print("Shock applied to Governance at t=60")

    # Policy intervention: AI boost after t=40
    if policy_AI_boost and t > 40:
        rho_eff = rho * 1.5
    else:
        rho_eff = rho

    # Compute sector updates
    for i in range(n):
        inter_sector = sum(beta[i, j] * S[t-1, j] for j in range(n))
        decay = -alpha[i] * S[t-1, i]
        ai_benefit = gamma[i] * A[t-1]
        ai_risk = -delta[i] * S[t-1, i] * A[t-1]

        # Policy: Shield agriculture from AI fragility
        if policy_shield_agriculture and i == 1:
            ai_risk *= 0.5

        dS = (decay + inter_sector + ai_benefit + ai_risk) * dt
        S[t, i] = max(S[t-1, i] + dS, 0)

    # Compute AI update
    feedback_cost = np.dot(theta, S[t-1])
    dA = (rho_eff - kappa * A[t-1] - feedback_cost) * dt
    A[t] = max(A[t-1] + dA, 0)

# Plotting
plt.figure(figsize=(12, 6))
for i in range(n):
    plt.plot(time, S[:, i], label=sectors[i])
plt.plot(time, A, label="AI Capability", linestyle='--', color='black')
plt.title("Sector and AI Dynamics Under Shocks and Policy")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('superintelligence_ode_shock_v1.png', dpi=300)
plt.close()
