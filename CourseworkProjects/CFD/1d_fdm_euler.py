import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant, block_diag
from scipy.integrate import solve_ivp
from numpy.linalg import inv
from clawpack import pyclaw, riemann
from clawpack.riemann.euler_with_efix_1D_constants import density, momentum, energy, num_eqn

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def D_operator_periodic(N,L,R,a):
  first_row = np.zeros(N); first_row[0:L+R+1] = a; first_row = np.roll(first_row,-L)
  return np.array(circulant(first_row)).transpose()

def Euler1D(t, Q, Dminus, Dplus):

  c = getSpeedofSound(Q)
  Omega = getEigenvalues(Q, c)
  X, invX = getEigenvectors(Q, c)
  Mminus, Mplus = getFluxJacobian(X, invX, Omega)
  return -(Mplus@Dplus + Mminus@Dminus)@Q

def getFluxJacobian(X, invX, Omega):

  OmegaPlus = 0.5 * (Omega + abs(np.array(Omega)))
  OmegaMinus = Omega - OmegaPlus
  
  Mplus = X@OmegaPlus@invX
  Mminus = X@OmegaMinus@invX

  return Mminus, Mplus

def getSpeedofSound(Q):

  c = []
  for i in range(0, int(len(Q)),3):
    c.append(np.sqrt(gamma * (gamma - 1) * (Q[i+2] - 0.5 * Q[i+1]**2/Q[i]) / Q[i]))
  return np.repeat(c,3)

def getEigenvectors(Q, c):

  X = []
  invX = []
  for i in range(0, int(len(Q)), 3):
    X0 = np.array([1, Q[i+1]/Q[i] - c[i], Q[i+2]/Q[i] + c[i]**2/gamma - Q[i+1]/Q[i] * c[i]])
    X1 = np.array([1, Q[i+1]/Q[i], 0.5 * (Q[i+1]/Q[i])**2])
    X2 = np.array([1, Q[i+1]/Q[i] + c[i], Q[i+2]/Q[i] + c[i]**2/gamma + Q[i+1]/Q[i] * c[i]])
    X.append([X0, X1, X2])

  invX = inv(X)
  return block_diag(*X).T, block_diag(*invX).T

def getEigenvalues(Q, c):
  eigs = []
  
  for i in range(0, int(len(Q)), 3):             
    eigs.append(np.diag([Q[i+1]/Q[i] - c[i], Q[i+1]/Q[i], Q[i+1]/Q[i] + c[i]]))
    
  return block_diag(*eigs) 

# General integrator function
def Integrator(periodic,operator,problem,L,T,Nx,Nt,u0, rho0, p0):

  # Initialize spatial domain
  x  = np.linspace(0, L, Nx, endpoint=(not periodic))
  dx = x[1] - x[0]
  p0s = np.zeros_like(x)
  for i in range(len(p0s)):
    p0s[i] = p0(x[i])
  E0s = p0s / (rho0*(gamma-1)) + 0.5 * u0**2

  # Initialize temporal domain
  t  = np.linspace(0, T, Nt, endpoint=True)

  Q0 = []
  q0 = np.full_like(E0s, rho0)
  q1 = np.full_like(E0s, rho0 * u0)
  q2 = rho0 * E0s

  for i in range(len(q0)):
    Q0.append(q0[i])
    Q0.append(q1[i])
    Q0.append(q2[i])

  # Construct spatial matrix operator
  match (operator,periodic):
    case('Flux-Splitting', True):
        Dplus = D_operator_periodic(3*Nx, 9, 3, [-1/(12*dx), 0, 0, 1/(2*dx), 0, 0, -3/(2*dx), 0, 0, 5/(6*dx), 0, 0, 1/(4*dx)])
        Dminus = D_operator_periodic(3*Nx, 3, 9, [-1/(4*dx), 0, 0, -5/(6*dx), 0, 0, 3/(2*dx), 0, 0, -1/(2*dx), 0, 0, 1/(12*dx)])
    case _:
      raise Exception("The %s operator '%s' is not yet implement!" % ('periodic' if periodic else 'non-periodic', operator))

  # Solve and return solutions!
  match problem:
    case 'Euler1D':
      # Solve initial value problem; see documentation at:
      # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
      sol = solve_ivp(Euler1D, [0, T], Q0, args=(Dminus, Dplus,), t_eval=t, first_step=1e-6,rtol=1.0e-6, atol=1.0e-6)
      # Transpose solution vector so that U has the format (Nt x Nx)
      Q = sol.y.transpose()
      # Return outputs
      return t, Q
    case _:
      raise Exception("The case '%s' is not yet implement!" % problem)

def modifiedWaveNumbers(operator, kdx):
  match(operator):
    case('Flux-Splitting'):
      l2r = (10 - 15 * np.cos(kdx) + 21 * 1j * np.sin(kdx) + 6 * np.cos(2 * kdx) - 6 * 1j * np.sin(2 * kdx) - np.cos(3 * kdx) + 1j * np.sin(3 * kdx)) / (12 * 1j) # left to right
      r2l = (np.cos(3 * kdx) + 1j * np.sin(3 * kdx) - 6 * np.cos(2 * kdx) - 6 * 1j * np.sin(2 * kdx) + 15 * np.cos(kdx) + 21 * 1j * np.sin(kdx) - 10) / (12 * 1j) # right to left
      return l2r, r2l
    
# Solution comparison
def pyclawsol(L, Nx, rho0, u0, p0):

    rs = riemann.euler_1D_py.euler_roe_1D

    solver = pyclaw.ClawSolver1D(rs)

    solver.kernel_language = 'Python'

    solver.bc_lower[0]=pyclaw.BC.periodic
    solver.bc_upper[0]=pyclaw.BC.periodic

    x = pyclaw.Dimension(0, L, Nx)
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain,num_eqn)

    state.problem_data['gamma'] = gamma
    state.problem_data['gamma1'] = gamma - 1.
    state.problem_data['efix'] = False

    xs = np.linspace(0, L, Nx, endpoint=False)
    p0s = np.zeros_like(xs)

    for i in range(len(xs)):
        p0s[i] = p0(xs[i])

    state.q[density,:] = rho0
    state.q[momentum,:] = rho0 * u0
    velocity = state.q[momentum,:]/state.q[density,:]
    pressure = p0s
    state.q[energy,:] = pressure/(gamma - 1.) + 0.5 * state.q[density,:] * velocity**2

    claw = pyclaw.Controller()
    claw.tfinal = 0.02
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.num_output_times = 5
    claw.keep_copy = True

    claw.run()
    
    rhos = []
    rhous = []
    rhoEs = []

    for i in range(0, 6):
        rhos.append(claw.frames[i].q[0])
        rhous.append(claw.frames[i].q[1])
        rhoEs.append(claw.frames[i].q[2])

    return rhos, rhous, rhoEs

# Problem 8

L = 10
Nx = 150
gamma = 1.4
rho0 = 1.225
u0 = 100
pinf = 101325
p0 = lambda x: pinf * (1 + 1/10 * np.exp(-10 * (x - L/2)**2))

x = np.linspace(0, L, Nx, endpoint=False)
T = 0.02
Nt = 6
t = np.linspace(0, T, Nt, endpoint=True)

kdx = np.linspace(0, np.pi, endpoint=True)

l2r, r2l = modifiedWaveNumbers('Flux-Splitting', kdx)
[t, Q] = Integrator(True, 'Flux-Splitting', 'Euler1D', L, T, Nx, Nt, u0, rho0, p0)

# Problem 9

clawrhos, clawrhous, clawrhoEs = pyclawsol(L, Nx, rho0, u0, p0)

clawus = [q1 / q0 for q1,q0 in zip(clawrhous, clawrhos)]
clawps = [(gamma - 1) * (q2 - 0.5 * q1**2 / q0) for q0, q1, q2 in zip(clawrhos, clawrhous, clawrhoEs)]

#Plotting

plt.figure(1)
plt.title('Left to Right Upwind Fourier Error Analysis')
plt.plot(kdx, kdx, label='Exact Solution')
plt.plot(kdx, l2r.real, label=r'Re$(\kappa^*\Delta x)$')
plt.plot(kdx, l2r.imag, label=r'Im$(\kappa^*\Delta x)$')
plt.xlabel(r'$\kappa\Delta x$')
plt.ylabel(r'$\kappa^*\Delta x$')
plt.legend()
plt.savefig('kstarl2rplot.jpg', dpi=300)

plt.figure(2)
plt.title('Right to Left Upwind Fourier Error Analysis')
plt.plot(kdx, kdx, label='Exact Solution')
plt.plot(kdx, r2l.real, label=r'Re$(\kappa^*\Delta x)$')
plt.plot(kdx, r2l.imag, label=r'Im$(\kappa^*\Delta x)$')
plt.xlabel(r'$\kappa\Delta x$')
plt.ylabel(r'$\kappa^*\Delta x$')
plt.legend()
plt.savefig('kstarr2lplot.jpg', dpi=300)

rhoplot = plt.figure(3)
rhoplot_ax = rhoplot.add_subplot(111)

uplot = plt.figure(4)
uplot_ax = uplot.add_subplot(111)

pplot = plt.figure(5)
pplot_ax = pplot.add_subplot(111)

for i in range(0, Nt):
    rho = []
    u = []
    p = []
    for j in range(0, 3 * Nx, 3):
        rho.append(Q[i][j])
        u.append(Q[i][j+1] / Q[i][j])
        p.append((gamma - 1) * (Q[i][j+2] - 0.5 * Q[i][j] * (Q[i][j+1]/Q[i][j])**2))
    
    rhoplot_ax.plot(x, rho, label=f't = {t[i]} s')
    uplot_ax.plot(x, u, label=f't = {t[i]} s')
    pplot_ax.plot(x, p, label=f't = {t[i]} s')

rhoplot_ax.legend()
uplot_ax.legend()
pplot_ax.legend()

rhoplot_ax.set_title('Density Variation')
rhoplot_ax.set_xlabel(r'$x$ [m]')
rhoplot_ax.set_ylabel(r'$\rho$ [kg/m$^3$]')

uplot_ax.set_title('Velocity Variation')
uplot_ax.set_xlabel(r'$x$ [m]')
uplot_ax.set_ylabel(r'$u$ [m/s]')

pplot_ax.set_title('Pressure Variation')
pplot_ax.set_xlabel(r'$x$ [m]')
pplot_ax.set_ylabel(r'$p$ [Pa]')

rhoplot.savefig('rhoplot.jpg', dpi=300)
uplot.savefig('uplot.jpg', dpi=300)
pplot.savefig('pplot.jpg', dpi=300)

clawrhoplot = plt.figure(6)
clawrhoplot_ax = clawrhoplot.add_subplot(111)

clawuplot = plt.figure(7)
clawuplot_ax = clawuplot.add_subplot(111)

clawpplot = plt.figure(8)
clawpplot_ax = clawpplot.add_subplot(111)

for i in range(0, len(clawrhos)):
    clawrhoplot_ax.plot(x, clawrhos[i], label=f't = {t[i]} s')
    clawuplot_ax.plot(x, clawus[i], label=f't = {t[i]} s')
    clawpplot_ax.plot(x, clawps[i], label=f't = {t[i]} s')

clawrhoplot_ax.set_title('PyClaw Density Variation')
clawrhoplot_ax.set_xlabel('x [m]')
clawrhoplot_ax.set_ylabel(r'$\rho$ [kg/m$^3$]')

clawuplot_ax.set_title('PyClaw Velocity Variation')
clawuplot_ax.set_xlabel('x [m]')
clawuplot_ax.set_ylabel('u [m/s]')

clawpplot_ax.set_title('PyClaw Pressure Variation')
clawpplot_ax.set_xlabel('x [m]')
clawpplot_ax.set_ylabel('p [Pa]')

clawrhoplot_ax.legend()
clawuplot_ax.legend()
clawpplot_ax.legend()

clawrhoplot.savefig('clawrhos.jpg', dpi=300)
clawuplot.savefig('clawus.jpg', dpi=300)
clawpplot.savefig('clawps.jpg', dpi=300)