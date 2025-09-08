import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
from scipy.optimize import fsolve
from functions import euler1DShockTubeSolver, shocktube_eq, getSpeeds, getTestTime
from decimal import Decimal

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Constants
gamma = 1.4
R = 287

def main():

    # Question 2

    rhoL = [1, 1]; uL = [0, -2]; pL = [1, 0.4]
    rhoR = [0.125, 1]; uR = [0, 2]; pR = [0.1, 0.4]
    tf = [0.25, 0.15]
    L1 = L4 = 0.5

    Nx = 501
    x = np.linspace(-L4, L1, Nx, endpoint=True)
    dx = x[1] - x[0]
    dt = 3.5e-4

    toroTest1 = np.genfromtxt('Test1_Toro.txt', skip_header=2)
    x1 = toroTest1[:,0]
    rho1 = toroTest1[:,3]
    u1 = toroTest1[:,4]
    p1 = toroTest1[:,5]

    toroTest1 = np.genfromtxt('Test2_Toro.txt', skip_header=2)
    x2 = toroTest1[:,0]
    rho2 = toroTest1[:,3]
    u2 = toroTest1[:,4]
    p2 = toroTest1[:,5]

    L1BC, L4BC = 'farfield', 'farfield'

    solver = ['Exact Riemann', 'Roe Approximate']

    style = ['solid', 'dotted']

    for i in range(len(rhoL)): 

        fig, (ax1, ax2, ax3), = plt.subplots(3, 1, figsize = (6,6), sharex=True)

        if i == 0:
            ax1.plot(x1, rho1, label='Toro Solution')
            ax2.plot(x1, u1, label='Toro Solution')
            ax3.plot(x1, p1, label='Toro Solution')
        else:
            ax1.plot(x2, rho2, label='Toro Solution')
            ax2.plot(x2, u2, label='Toro Solution')
            ax3.plot(x2, p2, label='Toro Solution')

        for j in range(len(solver)):

            if j == 1:
                dt = 9e-6

            t = np.arange(0, tf[i] + dt, dt)

            q1 = [rhoR[i], uR[i], pR[i]]
            q4 = [rhoL[i], uL[i], pL[i]]
            rho, rhou, rhoE = euler1DShockTubeSolver(q1, q4, x, dx, t, dt, L1BC, L4BC, solver[j])

            rhotf = rho[1:,-1]
            utf = rhou[1:,-1] / rhotf
            ptf = (gamma - 1) * (rhoE[1:,-1] - 0.5 * rhotf * utf**2)

            ax1.plot(x, rhotf, label=solver[j], linestyle=style[j])
            ax1.set_ylabel(r'$\rho$ [kgm$^{-3}$]')
            ax1.legend()

            ax2.plot(x, utf, label=solver[j], linestyle=style[j])
            ax2.set_ylabel(r'$u$ [m/s]')
            ax2.legend()

            ax3.plot(x, ptf, label=solver[j], linestyle=style[j])
            ax3.set_ylabel(r'$p$ [Pa]')
            ax3.legend()

            plt.xlabel(r'$x$ [m]')
            plt.tight_layout()

        if j == 0:
            fig.savefig('torotest' + str(i+1) + '.jpeg', dpi=300)
        else:
            fig.savefig('torotestcomp' + str(i+1) + '.jpeg', dpi=300)

    # Question 3

    # Geometry and test ICs
    L1 = 19; L4 = 1; Lmodel = L1/2
    p1 = 0.1 * 1e5; p4 = 100*1e5
    T1 = T4 = 293
    c1 = np.sqrt(gamma * R * T1); c4 = np.sqrt(gamma * R * T4)

    rho1 = p1 / (R * T1)
    rho4 = p4 / (R * T4)

    q4 = [rho4, 0, p4]
    q1 = [rho1, 0, p1]

    # Time and space discretizations
    Nx = [21, 41, 81, 161, 321, 641]
    Nt = [301, 601, 1201, 2401, 4801, 9601]
    tf = 1e-1

    # Boundary conditions
    L1BC = L4BC = 'solid wall'

    Ms_exact = fsolve(shocktube_eq, 3, args=(c4, c1, p4, p1,))[0] # exact solution shock Mach number
    Ms_sim_exact = []
    Ms_sim_roe = []
    dxs = []
    dts = []

    # Plotting parameters
    extent = -L4, L1, 0, tf * 1e3
    interp = 'bilinear'
    origin = 'lower'
    figsize = (8,3)
    labels = [r'$\rho$ [kgm$^{-3}$]', r'$u$ [m/s]', r'$p$ [bar]', r'$T$ [K]']
    norms = [colors.LogNorm(), None, colors.LogNorm(), colors.LogNorm()]

    rhoExact = []; uexact = []; pexact = []; rhoRoe = []; uRoe = []; pRoe = []

    for k in range(len(Nx)):

        x = np.linspace(-L4, L1, Nx[k], endpoint=True)
        dx = x[1] - x[0]
        dxs.append(dx)
        t = np.linspace(0, tf, Nt[k], endpoint=True)
        dt = t[1] - t[0]
        dts.append(dt)
        ts = np.where([Decimal(str(tt)) % Decimal('0.01') < 1e-8 for tt in t])[0] # get solutions for t = 0, 0.01, ..., 0.1

        print('')
        print(f'For Nx = {Nx[k]}, Nt = {Nt[k]}')

        for i in range(len(solver)):

            start = time.time()

            rho, rhou, rhoE = euler1DShockTubeSolver(q1, q4, x, dx, t, dt, L1BC, L4BC, solver[i])

            end = time.time()

            elapsed = end - start
            print(solver[i] + ' solver wall time: %.5f seconds' %elapsed)

            rho = rho[1:,:]
            u = rhou[1:,:] / rho
            p = (gamma - 1) * (rhoE[1:,:] - 0.5 * rho * u**2)
            T = p / (rho * R)
            p = p * 1e-5

            us, ucs = getSpeeds(rho, rhou, rhoE, x, t)
            Ms = us/c1
            err = abs(Ms_exact - Ms) / Ms_exact * 100
            print(solver[i] + f' shock Mach number: {Ms:.4g}, percent error: {err:.4g}')

            if i == 0:
                Ms_sim_exact.append(Ms)
            else:
                Ms_sim_roe.append(Ms)

            testtime = getTestTime(Lmodel, us, ucs)
            print(solver[i] + f' model test time: {testtime*1000:.5g} ms')

            data = [rho.T, u.T, p.T, T.T]

            if k == len(Nx) - 1:

                for j, (d, label, norm) in enumerate(zip(data, labels, norms), i):
                    fig, ax = plt.subplots(figsize=figsize)
                    contour = ax.imshow(d, cmap='viridis', interpolation=interp, extent=extent, origin=origin, aspect='auto', norm=norm)
                    ax.set_xlabel(r'$x$ [m]')
                    ax.set_ylabel(r'$t$ [ms]')
                    colorbar = fig.colorbar(contour, ax=ax)
                    colorbar.set_label(label)
                    plt.tight_layout()
                    fig.savefig(f'figure{i}{j}.jpeg', dpi=300)

            if k == len(Nx) - 1 and i == 0:
                rhoExact.append(rho)
                uexact.append(u)
                pexact.append(p)
            elif k == len(Nx) - 1 and i == 1:
                rhoRoe.append(rho)
                uRoe.append(u)
                pRoe.append(p)

            plt.close()

    for j in range(len(ts)):

        fig, axs = plt.subplots(3, 1, figsize = (6,6), sharex=True)

        axs[0].plot(x, rhoExact[0][:,ts[j]], label=solver[0], linestyle=style[0])
        axs[0].plot(x, rhoRoe[0][:,ts[j]], label=solver[1], linestyle=style[1])
        axs[0].set_ylabel(r'$\rho(x, t = %.2f)$ [kg/m$^{-3}$]'% round(t[ts[j]],2))
        axs[0].set_yscale('log')
        axs[0].legend()

        axs[1].plot(x, uexact[0][:,ts[j]], label=solver[0],linestyle=style[0])
        axs[1].plot(x, uRoe[0][:,ts[j]], label=solver[1],linestyle=style[1])
        axs[1].set_ylabel(r'$u(x, t = %.2f)$ [m/s]' % round(t[ts[j]],2))
        axs[1].legend()

        axs[2].plot(x, pexact[0][:,ts[j]], label=solver[0],linestyle=style[0])
        axs[2].plot(x, pRoe[0][:,ts[j]], label=solver[1],linestyle=style[1])
        axs[2].set_ylabel(r'$p(x,t = %.2f)$ [bar]' % round(t[ts[j]],2))
        axs[2].set_yscale('log')
        axs[2].legend()
        
        plt.xlabel(r'$x$ [m]')   
        plt.tight_layout()

        fig.savefig('shocktubesolvercomp' + str(round(t[ts[j]],2)) + '.jpeg', dpi=300)

        plt.close()

if __name__ == '__main__':
    main()