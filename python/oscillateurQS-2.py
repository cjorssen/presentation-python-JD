import numpy as np              
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

tfin = 15e-3                      # instant final de la simulation
dt = 0.01e-3                      # pas temporel
Vsat = 15                         # potentiel de saturation de l'ALI

T0 = tfin / 10
f0 = 1 / T0
omega0 = 2 * np.pi * f0
H0 = 1 / 3
Q = 1 / 3
G = 1.09 / H0

# Conditions Initiales
t = [0.0]
e = [0.01]   # Entrée de l'étage d'amplification / sortie du filtre ; 0.01 simule le bruit
s = [0.0]    # Sortie de l'étage d'amplification / entrée du filtre
dsdt = [0.0]   # Dérivée de s
dedt = [0.01]  # Dérivée de e ; 0.01 simule le bruit en entree

mode = [0] # 0 = linéaire, +1 = saturation positive, -1 = saturation négative

CIs = [[t[0], e[0], dedt[0], s[0], dsdt[0]]]
types_ED = [mode[0]]

# Resolution de l'équation différentielle par la méthode d'Euler
while t[-1] < tfin:
    dedt.append((omega0 * dsdt[-1] - omega0 / Q * dedt[-1] - e[-1] * omega0**2) * dt + dedt[-1])
    e.append(dedt[-1] * dt + e[-1])
    if G * e[-1] < -Vsat:
        s.append(-Vsat)
        mode.append(-1)
    elif G * e[-1] > Vsat:
        s.append(Vsat)
        mode.append(1)
    else:
        s.append(G * e[-1])
        mode.append(0)
    dsdt.append((s[-1] - s[-2]) / dt)
    
    if mode[-1] != mode[-2]:
        CIs.append([t[-1], e[-1], dedt[-1], s[-1], dsdt[-1]])
        types_ED.append(mode[-1])
    
    t.append(t[-1] + dt)

e = np.asarray(e)
dedt = np.asarray(dedt)
s = np.asarray(s)
t = np.asarray(t)
mode = np.asarray(mode)

e_ED = []
dedt_ED = []
t_ED = []

def ED_mode0(X, t, Q, G, H0):
    e, dedt = X
    return [dedt,
            - (omega0 * (1 - G * H0) * dedt / float(Q)) - (omega0**2 * e)]

def ED_mode_not0(X, t, Q, G, H0):
    e, dedt = X
    return [dedt,
            - (omega0 * dedt / float(Q)) - (omega0**2 * e)]
    
for i, CI in enumerate(CIs):
    t_ED.append(np.linspace(CI[0], CI[0] + 2 * T0, 100))

    if types_ED[i] == 0:
        sol = odeint(ED_mode0, [CI[1], CI[2]], t_ED[-1], args = (Q, G, H0))
    else:
        sol = odeint(ED_mode_not0, [CI[1], CI[2]], t_ED[-1], args = (Q, G, H0))

    e_ED.append(sol[:,0])
    dedt_ED.append(sol[:,1])        

plt.subplot(3,1,1)   # Sous-figure 1: évolution temporelle de e
plt.plot(t, e)
plt.plot(t[mode == 1], e[mode == 1], 'r.')
plt.plot(t[mode == -1], e[mode == -1], 'g.')

plt.xlim([0, np.max(t_ED)])
plt.ylim([-10, 10])

for i, CI in enumerate(CIs):
    if types_ED[i] == 0:
        style = 'C0--'
    elif types_ED[i] == 1:
        style = 'r--'
    else:
        style = 'g--'
        
    plt.plot(t_ED[i], e_ED[i], style, linewidth = .5)

plt.yticks([-Vsat / G, Vsat / G])
plt.grid(which = 'both')

plt.title('f0 = {0:.3f} Hz, Q = {1:.3f}, H0 = {2:.2f}, G = {3:.3f}, Vsat = {4:.2f} V, Vsat / G = {5:.3f} V'.format(f0, Q, H0, G, Vsat, Vsat / G))

plt.xlabel("t (s)")
plt.ylabel("e (V)")

plt.subplot(3,1,2)   # Sous-figure 2 : évolution temporelle de s
plt.plot(t, s)
plt.plot(t[mode == 1], s[mode == 1], 'r.')
plt.plot(t[mode == -1], s[mode == -1], 'g.')
plt.xlim([0, np.max(t_ED)])
plt.ylim([-20, 20])

plt.yticks([-Vsat, Vsat])
plt.grid(which = 'both')

plt.xlabel("t (s)")
plt.ylabel("s (V)")

plt.subplot(3,1,3)   # Sous-figure 3 : Portrait de phase
plt.plot(e, dedt)
plt.plot(e[mode == 1], dedt[mode ==1], 'r.')
plt.plot(e[mode == -1], dedt[mode ==-1], 'g.')

plt.xticks([-Vsat / G, 0, Vsat / G])
plt.grid(which = 'both')

plt.xlabel("e (V)")
plt.ylabel("de/dt (V/s)")
plt.title("Plan de phase")

plt.tight_layout()   # Pour ajuster les espaces autour des sous-figures
plt.show()
