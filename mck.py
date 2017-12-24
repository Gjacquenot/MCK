# -*- coding: utf-8 -*-

# 1
# Gilberto A. Ortiz, Diego A. Alvarez, Daniel Bedoya-Ruiz.
# "Identification of Bouc-Wen type models using multi-objective optimization algorithms".
# Computers & Structures. Vol. 114-115. pp. 121-132. 2013.
# https://sourceforge.net/projects/boucwenbabernoo/?source=typ_redirect

# 2
# Introduction of D parameter
# Dynamic properties of the hysteretic Bouc-Wen model
# doi:10.1016/j.sysconle.2006.09.001
# ftp://ftp.ecn.purdue.edu/ayhan/Bismarck/Dynamic%20properties%20of%20the%20hysteretic%20Bouc-Wen%20model.pdf

# 3
# https://github.com/rock-control/control-hysteresis_model/tree/master/coupling_calibration

# 4
# https://icesd.hi.is/wp-content/uploads/2017/06/AID_22.pdf

import numpy as np

class MCK():
    def __init__(self, **kwargs):
        # m x'' + c x' + k x  = Fext(t)
        #   x'' + c/m x' + k/m x  = Fext(t)/m
        #
        # [x' ]   [[0    ,    1]]   [x ]   [         0 ]
        # |   | = ||           || * |  | + |           |
        # [x'']   [[-k/m , -c/m]]   [x']   [ Fext(t)/m ]
        self.m = kwargs.get('m', 1)
        self.c = kwargs.get('c', 0)
        self.k = kwargs.get('k', 1)
        self.names = ['x', 'v']
        self.n_states = len(self.names)
        if self.m < 0.0:
            raise Exception

    def derive(self, t, x, **kwargs):
        fext = kwargs.get('fext', 0)
        df = [0.0, 0.0]
        df[0] = x[1]
        df[1] = (- self.k * x[0] - self.c * x[1] + fext) / self.m
        return np.array(df)


class WB():
    def __init__(self, **kwargs):
        """
            - disp      = x[0];     % system displacement
            - vel       = x[1];     % system velocity
            - zeta      = x[2];     % hysteretic component
            - eps       = x[3];     % hysteretic energy
        """
        self.m = kwargs.get('m', 1)
        self.c = kwargs.get('c', 0)

        # α=0.5; k=1; D=1; A=1; β=0.5; γ= −1.5; n=2
        self.alpha = kwargs.get('alpha', 0.5)
        self.k = kwargs.get('k', 1)
        self.D = kwargs.get('D', 1)
        self.A = kwargs.get('A', 1)
        self.beta = kwargs.get('beta', 0.5)
        self.gamma = kwargs.get('gamma', -1.5)
        self.n = kwargs.get('n', 2)

        self.names = ['x', 'v', 'zeta', 'eps']
        self.n_states = len(self.names)


    def derive(self, t, x, **kwargs):

        # m x'' + c x' + kHys(x,t) = Fext(t)
        #
        # x'' = 1/m( Fext(t) - c x' - kHys(x,t))
        #
        # kHys(x,t) = alpha * k * x + (1 - alpha) * D * k * z(t)
        # z'(t) = + 1/D * (A * x'(t) - beta * abs(x'(t)) * abs(z(t))**(n-1) * z(t) - gamma * x'(t) * abs(z(t))**n)

        fext = kwargs.get('fext', 0)
        dxdt = [0.0, 0.0, 0.0, 0.0]
        w0 = np.sqrt(self.k / self.m);  # Natural frequency (rad/s)
        # x1
        dxdt[0] = x[1]
        # exci - 2*xi*w0*x1 - alpha*(w0^2)*x0 - (1-alpha)*(w0^2)*x2;
        dxdt[1] = (fext - self.c * x[1] - self.k * (self.alpha * x[0] + (1 - self.alpha) * self.D * x[2])) / self.m
        # h*( x1 - nueps*(beta*abs(x1)*(abs(z)^(n-1))*z + gamma*x1*(abs(z)^n) ) ) / etaeps;
        dxdt[2] = 1/self.D * (self.A * x[1] - self.beta * np.abs(x[1]) * np.abs(x[2]) ** (self.n - 1) * x[2] - self.gamma * x[1] * np.abs(x[2]) ** self.n)
        # (1-alpha)*(w0^2)*x1*z;
        dxdt[3] = (1-self.alpha) * self.k / self.m * x[1] * x[2]
        return np.array(dxdt)


class Integrator():
    def __init__(self, system, external_data={}):
        self.system = system
        self.external_data = external_data

    def get_external_data(self, t):
        return {k: np.interp(t, self.external_data[k][:, 0],
                                self.external_data[k][:, 1]) for k in self.external_data}

    def integ(self, x0, dt=1, t_end=1, t_start=0.0):
        integration_scheme = self.rk44
        self.dt = dt
        if len(x0) < self.system.n_states:
            x0 += [0.0] * (self.system.n_states - len(x0))
        time_steps = np.arange(t_start + dt, t_end + dt, dt)
        states = np.zeros((len(time_steps) + 1, len(x0) + 1))
        states[0, 0] = t_start
        states[0, 1:] = x0
        for i, t in enumerate(time_steps):
            x0 = integration_scheme(t=t, x0=x0)
            states[i + 1, 0] = t
            states[i + 1, 1:] = x0
        states = np.rec.fromrecords(states, names = 't, ' + ', '.join(self.system.names))
        return states

    def euler(self, t, x0):
        dt = self.dt
        F = self.system.derive
        ext_t0 = self.get_external_data(t)
        x1 = x0 + dt * F(t, x0, **ext_t0)
        return x1

    def rk44(self, t, x0):
        """
        explicit runge kutta 4*4
        """
        dt = self.dt
        F = self.system.derive
        ext_t0 = self.get_external_data(t)
        ext_t1 = self.get_external_data(t + dt/2.0)
        ext_t2 = self.get_external_data(t + dt)
        k1 = dt * F(t, x0, **ext_t0)
        k2 = dt * F(t + dt/2.0, x0 + k1/2.0, **ext_t1)
        k3 = dt * F(t + dt/2.0, x0 + k2/2.0, **ext_t1)
        k4 = dt * F(t + dt, x0 + k3, **ext_t2)
        x1 = x0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4)/6.0  # + error O(dt^5)
        return x1


def plot_states(states, **kwargs):
    import matplotlib.pyplot as plt
    external_data = kwargs.get('external_data',{})

    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    for n in states.dtype.names[1:]:
        ax.plot(states['t'], states[n], label=n)
    for k in external_data:
        ax.plot(external_data[k][:, 0], external_data[k][:, 1], label=k)
    ax.set(xlabel='time (s)', ylabel='states', title='State variables')
    ax.grid()
    ax.legend(loc='upper right')
    # fig.savefig("test.png")
    plt.show()


def plot_f_wrt_x(states, **kwargs):
    import matplotlib.pyplot as plt
    external_data = kwargs.get('external_data',{})
    fig, ax = plt.subplots()
    n = len(states['x'])
    ax.plot(external_data['fext'][:,1], states['x'], label='x = F')
    ax.plot(external_data['fext'][(2*n//5):n,1], states['x'][(2*n//5):n])
    ax.set(xlabel='F (N)', ylabel='x (m)', title='State variables')
    ax.grid()
    ax.legend(loc='upper right')
    # fig.savefig("test.png")
    plt.show()


def demo(system=MCK, **kwargs):
    mck = system(m=20, c=1)
    t_stop = kwargs.get('t_stop', 1000)
    dt = kwargs.get('dt', 0.1)
    initial_states = kwargs.get('initial_states', [0.0, 0.0])
    #
    t = np.arange(0, t_stop + dt, dt)
    d = 1 - np.arange(0, t_stop + dt, dt)/t_stop
    f = np.sin(2*np.pi*t/(t_stop/50)) * d
    tf  = np.concatenate((np.vstack(t), np.vstack(f)),axis=1)
    integrator = Integrator(mck, external_data={'fext': tf})
    states = integrator.integ(x0=initial_states, t_end=t_stop, dt=dt)
    #
    plot_states(states, external_data={'fext': tf})
    plot_f_wrt_x(states, external_data={'fext': tf})


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    pa = parser.add_argument
    pa('-s', '--system', help='Name of the system to simulate (default: will display a demo)', default='')
    pa('--dt', type=float, help='', default=0.1)
    return parser


def main(cli=None):
    parser = get_parser()
    args = parser.parse_args(cli)
    if args.system:
        if args.system.lower() == 'wb':
            demo(system=WB)
        elif args.system.lower() == 'mck':
            demo(system=MCK)
    else:
        demo()


if __name__=='__main__':
    main()