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
        # m x'' + c x' + k x = Fext(t)
        #   x'' + c/m x' + k/m x = Fext(t)/m
        #
        # [x' ]   [[0    ,    1]]   [x ]   [         0 ]
        # |   | =  |           |  * |  | + |           |
        # [x'']   [[-k/m , -c/m]]   [x']   [ Fext(t)/m ]
        self.m = kwargs.get('m', 1)
        self.c = kwargs.get('c', 0)
        self.k = kwargs.get('k', 1)
        self.names = ['x', 'v']
        self.n_states = len(self.names)
        if self.m < 0.0:
            raise Exception

    def __repr__(self):
        return 'M={0} C={1} K={2} T={3:4f} s'.format(self.m, self.c, self.k, self.get_period())

    def get_period(self):
        # omega = 2 * np.pi * f = 2 * np.pi / t = np.sqrt(self.k/self.m)
        return 2 * np.pi / np.sqrt(self.k/self.m)

    def derive(self, _, x, **kwargs):
        fext = kwargs.get('fext', 0)
        df = [0.0, 0.0]
        df[0] = x[1]
        df[1] = (- self.k * x[0] - self.c * x[1] + fext) / self.m
        return np.array(df)

    def analytical_solution(self, x0, t):
        """
        return analytical solution when Fext = 0
        """
        # α r**2 + β r + γ=0.
        # m x'' + c x' + k x
        m = self.m
        c = self.c
        k = self.k
        # delta can be complex
        delta = (c**2 - 4 * m * k)**0.5
        x1 = -c - delta /(2*m)
        x2 = -c + delta /(2*m)
        raise NotImplementedError


class WB():
    def __init__(self, **kwargs):
        """
        Bouc Wen without degradation nor pinching
            - disp  = x[0];  # system displacement
            - vel   = x[1];  # system velocity
            - zeta  = x[2];  # hysteretic component
            - eps   = x[3];  # hysteretic energy
        """
        self.m = kwargs.get('m', 1)
        self.c = kwargs.get('c', 0)

        # α=0.5; k=1; D=1; A=1; β=0.5; γ= -1.5; n=2
        self.alpha = kwargs.get('alpha', 0.5)
        self.k = kwargs.get('k', 1)
        self.D = kwargs.get('D', 1)
        self.A = kwargs.get('A', 1)
        self.beta = kwargs.get('beta', 0.5)
        self.gamma = kwargs.get('gamma', -1.5)
        self.n = kwargs.get('n', 2)

        self.names = ['x', 'v', 'zeta', 'eps']
        self.n_states = len(self.names)

    def __repr__(self):
        return 'M={0} C={1} K={2} α={3} D={4} A={5} β={6} γ={7} n={8}'.format(
                    self.m, self.c, self.k,
                    self.alpha, self.D, self.A,
                    self.beta, self.gamma, self.n)

    def derive(self, t, x, **kwargs):

        # m x'' + c x' + kHys(x,t) = Fext(t)
        #
        # x'' = 1/m( Fext(t) - c x' - kHys(x,t))
        #
        # kHys(x,t) = alpha * k * x + (1 - alpha) * D * k * z(t)
        # z'(t) = + 1/D * (A * x'(t) - beta * abs(x'(t)) * abs(z(t))**(n-1) * z(t) - gamma * x'(t) * abs(z(t))**n)

        fext = kwargs.get('fext', 0)
        dxdt = [0.0, 0.0, 0.0, 0.0]
        w0 = np.sqrt(self.k / self.m)  # Natural frequency (rad/s)
        # x1
        dxdt[0] = x[1]
        # exci - 2*xi*w0*x1 - alpha*(w0^2)*x0 - (1-alpha)*(w0^2)*x2;
        dxdt[1] = (fext - self.c * x[1] - self.k * (self.alpha * x[0] + (1 - self.alpha) * self.D * x[2])) / self.m
        # h*( x1 - nueps*(beta*abs(x1)*(abs(z)^(n-1))*z + gamma*x1*(abs(z)^n) ) ) / etaeps;
        dxdt[2] = 1/self.D * (self.A * x[1] - self.beta * np.abs(x[1]) * np.abs(x[2]) ** (self.n - 1) * x[2] - self.gamma * x[1] * np.abs(x[2]) ** self.n)
        # (1-alpha)*(w0^2)*x1*z;
        dxdt[3] = (1-self.alpha) * self.k / self.m * x[1] * x[2]
        return np.array(dxdt)


class WB1():
    def __init__(self, **kwargs):
        """
        Bouc Wen with degradation no pinching
            - disp  = x[0];  # system displacement
            - vel   = x[1];  # system velocity
            - zeta  = x[2];  # hysteretic component
            - eps   = x[3];  # hysteretic energy
        """
        self.m = kwargs.get('m', 1)
        self.c = kwargs.get('c', 0)

        # α=0.5; k=1; D=1; A=1; β=0.5; γ= -1.5; n=2
        self.alpha = kwargs.get('alpha', 0.5)
        self.k = kwargs.get('k', 1)
        self.D = kwargs.get('D', 1)
        self.A = kwargs.get('A', 1)
        self.beta = kwargs.get('beta', 0.5)
        self.gamma = kwargs.get('gamma', -1.5)
        self.n = kwargs.get('n', 2)


        self.nu0 = kwargs.get('nu0', 0.0)               # strength degradation
        self.A0 = kwargs.get('A0', 0.0)                 # hysteresis amplitude
        self.eta0 = kwargs.get('eta0', 1.0)             # stiffness degradation
        self.delta_nu = kwargs.get('delta_nu', 0.0)     # strength degradation parameter
        self.delta_A = kwargs.get('delta_A', 0.0)       # control parameter of 'A' with respect to the energy
        self.delta_eta = kwargs.get('delta_eta', 0.0)   # stiffness degradation parameter

        self.names = ['x', 'v', 'zeta', 'eps']
        self.n_states = len(self.names)

    def __repr__(self):
        return 'M={0} C={1} K={2} α={3} D={4} A={5} β={6} γ={7} n={8}'.format(
                    self.m, self.c, self.k,
                    self.alpha, self.D, self.A,
                    self.beta, self.gamma, self.n)

    def derive(self, t, x, **kwargs):

        # m x'' + c x' + kHys(x,t) = Fext(t)
        #
        # x'' = 1/m( Fext(t) - c x' - kHys(x,t))
        #
        # kHys(x,t) = alpha * k * x + (1 - alpha) * D * k * z(t)
        # z'(t) = + 1/D * (A * x'(t) - beta * abs(x'(t)) * abs(z(t))**(n-1) * z(t) - gamma * x'(t) * abs(z(t))**n)

        fext = kwargs.get('fext', 0)
        dxdt = [0.0, 0.0, 0.0, 0.0]
        w0 = np.sqrt(self.k / self.m);              # Natural frequency (rad/s)
        # nueps, Aeps, etaeps                       # Degradation functions
        nueps = self.nu0  + self.delta_nu  * x[3]   # strength degradation function
        Aeps  = self.A0   - self.delta_A   * x[3]   # degradation function
        etaeps = self.eta0 + self.delta_eta * x[3]  # stiffness degradation function

        # x1
        dxdt[0] = x[1]
        # exci - 2*xi*w0*x1 - alpha*(w0^2)*x0 - (1-alpha)*(w0^2)*x2;
        dxdt[1] = (fext - self.c * x[1] - self.k * (self.alpha * x[0] + (1 - self.alpha) * self.D * x[2])) / self.m
        # h*( x1 - nueps*(beta*abs(x1)*(abs(z)^(n-1))*z + gamma*x1*(abs(z)^n) ) ) / etaeps;
        dxdt[2] = 1/self.D * (Aeps * x[1] - nueps * (self.beta * np.abs(x[1]) * np.abs(x[2]) ** (self.n-1) * x[2] + self.gamma * x[1] * np.abs(x[2]) ** self.n)) / etaeps
        # (1-alpha)*(w0^2)*x1*z;
        dxdt[3] = (1-self.alpha) * self.k / self.m * x[1] * x[2]
        return np.array(dxdt)


class VanDerPol():
    def __init__(self, **kwargs):
        """
        x'' = mu * (1 - x**2) * x' - x
        """
        self.mu = kwargs.get('m', 1)
        self.names = ['x', 'v']
        self.n_states = len(self.names)

    def __repr__(self):
        return 'Van der Pol -- Mu = {0}'.format(self.mu)

    def derive(self, _, x, **kwargs):
        df = [0.0, 0.0]
        df[0] = x[1]
        df[1] = self.mu * (1 - x[0]**2) * x[1] - x[0]
        return np.array(df)


class Integrator():
    def __init__(self, system, **kwargs):
        self.system = system
        self.verbose = kwargs.get('verbose', False)
        self.external_data = kwargs.get('external_data', {})
        self.dict_algorithm = {'euler': self.euler,
                               'rk22': self.rk22,
                               'rk44': self.rk44,
                               'rk45': self.rk45Fehlberg,
                               'rk45Fehlberg': self.rk45Fehlberg,
                               'rk45CashKarp': self.rk45CashKarp,
                               'rk45DormandPrince': self.rk45DormandPrince,
                               'implicit_euler': self.implicit_euler,
                               'bdf1': self.implicit_euler,
                               'scipy_rk23': self.scipy_rk23,
                               'scipy_rk45': self.scipy_rk45,
                               'scipy_radau': self.scipy_radau,
                               'scipy_bdf': self.scipy_bdf,
                               'scipy_lsoda': self.scipy_lsoda}

    @staticmethod
    def get_integration_algorithms():
        return ('euler', 'rk22', 'rk44', 'rk45', 'rk45Fehlberg', 'rk45CashKarp', 'rk45DormandPrince',
                'implicit_euler', 'bdf1', 'scipy_rk23', 'scipy_rk45',
                'scipy_radau', 'scipy_bdf', 'scipy_lsoda')

    def get_external_data(self, t):
        return {k: np.interp(t, self.external_data[k][:, 0],
                                self.external_data[k][:, 1]) for k in self.external_data}

    def timeit(method):
        import time
        def timed(*args, **kw):
            verbose = args[0].verbose
            if verbose:
                ts = time.time()
                result = method(*args, **kw)
                te = time.time()
                print('{0}  {1:2.2f} ms'.format(method.__name__, (te - ts) * 1000))
            else:
                result = method(*args, **kw)
            return result
        return timed

    @timeit
    def integ(self, x0, dt=1, t_end=1, t_start=0.0, algorithm='rk44'):
        integration_scheme = self.dict_algorithm[algorithm]
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
        """
        explicit euler scheme

          0 |
        ---------
            |   1

        https://en.wikipedia.org/wiki/Euler_method
        """
        dt = self.dt
        F = self.system.derive
        ext_t0 = self.get_external_data(t)
        k1 = F(t, x0, **ext_t0)
        x1 = x0 + dt * k1
        return x1

    def implicit_euler(self, t, x0):
        """
        Backward Euler method (BDF1)

            y_{{k+1}} = y_{k} + h f(t_{{k+1}}, y_{{k+1}})

          1 |  1
        --------
            |  1

        https://en.wikipedia.org/wiki/Backward_differentiation_formula
        """
        import scipy.optimize
        dt = self.dt
        F = self.system.derive
        ext_t1 = self.get_external_data(t + dt)
        def fun(sol):
            return x0 + dt * F(t + dt, sol, **ext_t1) - sol
        res = scipy.optimize.root(fun, x0)
        return res['x']


    def rk22(self, t, x0):
        """
        explicit runge kutta 2*2

          0 |
        1/2 | 1/2
        ---------------
            |   0 |   1
        """
        dt = self.dt
        F = self.system.derive
        ext_t0 = self.get_external_data(t)
        ext_t1 = self.get_external_data(t + dt / 2.0)
        k1 = F(t, x0, **ext_t0)
        k2 = F(t + dt / 2.0, x0 + dt / 2.0 * k1, **ext_t1)
        x1 = x0 + dt * k2
        return x1

    def rk44(self, t, x0):
        """
        explicit runge kutta 4*4

          0 |
        1/2 | 1/2
        1/2 |   0 | 1/2
          1 |   0 |   0 |   1
        ---------------------------
            | 1/6 | 1/3 | 1/3 | 1/6
        """
        dt = self.dt
        F = self.system.derive
        ext_t0 = self.get_external_data(t)
        ext_t1 = self.get_external_data(t + dt / 2.0)
        ext_t2 = self.get_external_data(t + dt)
        k1 = F(t, x0, **ext_t0)
        k2 = F(t + dt / 2.0, x0 + dt / 2.0 * k1, **ext_t1)
        k3 = F(t + dt / 2.0, x0 + dt / 2.0 * k2, **ext_t1)
        k4 = F(t + dt, x0 + dt * k3, **ext_t2)
        x1 = x0 + dt /6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)  # + error O(dt^5)
        return x1

    def rk45Fehlberg(self, t, x0):
        """
        explicit Runge–Kutta–Fehlberg method (with two methods of orders 5 and 4)

            0 |
          1/4 |        1/4
          3/8 |      3/32 |       9/32
        12/13 | 1932/2197 | −7200/2197 |  7296/2197
            1 |   439/216 |         −8 |   3680/513 |   -845/4104
          1/2 |     −8/27 |          2 | −3544/2565 |   1859/4104 | −11/40
        -------------------------------------------------------------------------
                   16/135 |          0 | 6656/12825 | 28561/56430 |  −9/50 | 2/55
                   25/216 |          0 |  1408/2565 |   2197/4104 |   −1/5 | 0

        Lower order solution is obtained from the second b line, the first one
        is used to estimate a higher order solution. The difference of the two
        gives an estimation of the integration error.
        """
        c2 = +1.0/4.0
        a21 = +1.0/4.0

        c3 = +3.0/8.0
        a31 = +3.0/32.0
        a32 = +9.0/32.0

        c4 = +12.0/13.0
        a41 = +1932.0/2197.0
        a42 = -7200.0/2197.0
        a43 = +7296.0/2197.0

        c5 = +1.0
        a51 = +439.0/216.0
        a52 = -8.0
        a53 = +3680.0/513.0
        a54 = -845.0/4104.0

        c6 = +1.0/2.0
        a61 = -8.0/27.0
        a62 = +2.0
        a63 = -3544.0/2565.0
        a64 = +1859.0/4104.0
        a65 = -11.0/40.0

        cy1 = +25.0/216.0
        cy3 = +1408.0/2565.0
        cy4 = +2197.0/4104.0
        cy5 = -1.0/5.0

        ce1 = +16.0/135.0-cy1
        ce3 = +6656.0/12825.0-cy3
        ce4 = +28561.0/56430.0-cy4
        ce5 = -9.0/50.0-cy5
        ce6 = +2.0/55.0

        dt = self.dt
        F = self.system.derive
        ext_t0 = self.get_external_data(t)
        ext_t2 = self.get_external_data(t + dt * c2)
        ext_t3 = self.get_external_data(t + dt * c3)
        ext_t4 = self.get_external_data(t + dt * c4)
        ext_t5 = self.get_external_data(t + dt * c5)
        ext_t6 = self.get_external_data(t + dt * c6)
        k1 = F(t, x0, **ext_t0)
        k2 = F(t + c2 * dt, x0 + dt * (a21 * k1), **ext_t2)
        k3 = F(t + c3 * dt, x0 + dt * (a31 * k1 + a32 * k2), **ext_t3)
        k4 = F(t + c4 * dt, x0 + dt * (a41 * k1 + a42 * k2 + a43 * k3), **ext_t4)
        k5 = F(t + c5 * dt, x0 + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), **ext_t5)
        x1 = x0 + dt * (cy1 * k1 + cy3 * k3 + cy4 * k4 + cy5 * k5)
        error = dt * (ce1 * k1 + ce3 * k3 + ce4 * k4 + ce5 * k5 + ce6 * k6)
        return x1

    def rk45CashKarp(self, t, x0):
        """
        explicit Runge–Kutta–Cash-Karp method (with two methods of orders 5 and 4)

        0    |
        1/5  |        1/5
        3/10 |       3/40 |    9/40
        3/5  |       3/10 |   -9/10 |         6/5
        1    |     -11/54 |     5/2 |      -70/27 |        35/27
        7/8  | 1631/55296 | 175/512 |   575/13824 | 44275/110592 |  253/4096 |
        --------------------------------------------------------------------------------
             | 37/378     |       0 |     250/621 |      125/594 |         0 | 512/1771
             | 2825/27648 |       0 | 18575/48384 |  13525/55296 | 277/14336 | 1/4
        """

        c2 = 1.0/5.0
        c3 = 3.0/10.0
        c4 = 3.0/5.0
        c5 = 1.0
        c6 = 7.0/8.0

        a21 = +1.0/5.0

        a31 = +3.0/40.0
        a32 = +9.0/40.0

        a41 = +3.0/10.0
        a42 = -9.0/10.0
        a43 = +6.0/5.0

        a51 = -11.0/54.0
        a52 = +5.0/2.0
        a53 = -70.0/27.0
        a54 = +35.0/27.0

        a61 = +1631.0/55296.0
        a62 = +175.0/512.0
        a63 = +575.0/13824.0
        a64 = +44275.0/110592.0
        a65 = +253.0/4096.0

        b1 = 37.0/378.0
        b2 = 0.0
        b3 = 250.0/621.0
        b4 = 125.0/594.0
        b5 = 0.0
        b6 = 512.0/1771.0

        b1p = 2825.0/27648.0
        b2p = 0.0
        b3p = 18575.0/48384.0
        b4p = 13525.0/55296.0
        b5p = 277.0/14336.0
        b6p = 1.0/4.0

        dt = self.dt
        F = self.system.derive

        ext_t1 = self.get_external_data(t)
        ext_t2 = self.get_external_data(t + dt * c2)
        ext_t3 = self.get_external_data(t + dt * c3)
        ext_t4 = self.get_external_data(t + dt * c4)
        ext_t5 = self.get_external_data(t + dt * c5)
        ext_t6 = self.get_external_data(t + dt * c6)

        k1 = F(t, x0, **ext_t1)
        k2 = F(t + c2 * dt, x0 + dt * (a21 * k1), **ext_t2)
        k3 = F(t + c3 * dt, x0 + dt * (a31 * k1 + a32 * k2), **ext_t3)
        k4 = F(t + c4 * dt, x0 + dt * (a41 * k1 + a42 * k2 + a43 * k3), **ext_t4)
        k5 = F(t + dt, x0 + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), **ext_t5)
        k6 = F(t + c6 * dt, x0 + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), **ext_t6)
        error = dt * abs((b1-b1p)*k1+(b3-b3p)*k3+(b4-b4p)*k4+(b5-b5p)*k5+
                         (b6-b6p)*k6)
        x1 = x0 + dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        return x1

    def rk45DormandPrince(self, t, x0):
        """

        Dormand, J. R. and P. J. Prince, “A family of embedded Runge-Kutta formulae,” J. Comp. Appl. Math., Vol. 6, 1980, pp. 19–26.

        Dormand and Prince chose the coefficients of their method to minimize the error of the fifth-order solution. This is the main difference with the Fehlberg method, which was constructed so that the fourth-order solution has a small error. For this reason, the Dormand–Prince method is more suitable when the higher-order solution is used to continue the integration, a practice known as local extrapolation (Shampine 1986; Hairer, Nørsett & Wanner 2008, pp. 178–179).

        0    |
        1/5  |        1/5
        3/10 |       3/40 | 9/40
        4/5  |      44/45 | -56/15      | 32/9
        8/9  | 19372/6561 | -25360/2187 | 64448/6561 | -212/729
        1    |  9017/3168 |     -355/33 | 46732/5247 | 49/176  |   -5103/18656
        1    |     35/384 |     0       | 500/1113   | 125/192 |    -2187/6784 | 11/84
        ----------------------------------------------------------------------------------------
                   35/384 |     0       | 500/1113   | 125/192 |    -2187/6784 | 11/84
               5179/57600 |     0       | 7571/16695 | 393/640 | -92097/339200 | 187/2100 | 1/40
        """
        a21 = +1.0/5.0

        a31 = +3.0/40.0
        a32 = +9.0/40.0

        a41 = +44.0/45.0
        a42 = -56.0/15.0
        a43 = +32.0/9.0

        a51 = +19372.0/6561.0
        a52 = -25360.0/2187.0
        a53 = +64448.0/6561.0
        a54 = -212.0/729.0

        a61 = +9017.0/3168.0
        a62 = -355.0/33.0
        a63 = +46732.0/5247.0
        a64 = +49.0/176.0
        a65 = -5103.0/18656.0

        a71 = +35.0/384.0
        a72 = 0.0
        a73 = +500.0/1113.0
        a74 = +125.0/192.0
        a75 = -2187.0/6784.0
        a76 = +11.0/84.0

        c2 = +1.0 / 5.0
        c3 = +3.0 / 10.0
        c4 = +4.0 / 5.0
        c5 = +8.0 / 9.0
        c6 = +1.0
        c7 = +1.0

        b1 = +35.0/384.0
        b2 = 0.0
        b3 = +500.0/1113.0
        b4 = +125.0/192.0
        b5 = -2187.0/6784.0
        b6 = +11.0/84.0
        b7 = 0.0

        b1p = +5179.0/57600.0
        b2p = +0.0
        b3p = +7571.0/16695.0
        b4p = +393.0/640.0
        b5p = -92097.0/339200.0
        b6p = +187.0/2100.0
        b7p = +1.0/40.0

        dt = self.dt
        F = self.system.derive

        ext_t1 = self.get_external_data(t)
        ext_t2 = self.get_external_data(t + dt * c2)
        ext_t3 = self.get_external_data(t + dt * c3)
        ext_t4 = self.get_external_data(t + dt * c4)
        ext_t5 = self.get_external_data(t + dt * c5)
        ext_t6 = self.get_external_data(t + dt * c6)

        k1 = F(t, x0, **ext_t1)
        k2 = F(t + c2 * dt, x0 + dt * (a21 * k1), **ext_t2)
        k3 = F(t + c3 * dt, x0 + dt * (a31 * k1 + a32 * k2), **ext_t3)
        k4 = F(t + c4 * dt, x0 + dt * (a41 * k1 + a42 * k2 + a43 * k3), **ext_t4)
        k5 = F(t + c5 * dt, x0 + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), **ext_t5)
        k6 = F(t +      dt, x0 + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), **ext_t6)
        k7 = F(t +      dt, x0 + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6), **ext_t6)

        error = dt * abs((b1 - b1p) * k1 +
                         (b3 - b3p) * k3 +
                         (b4 - b4p) * k4 +
                         (b5 - b5p) * k5 +
                         (b6 - b6p) * k6 +
                         (b7 - b7p) * k7)

        x1 = x0 + dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        return x1

    def _scipy_ode(self, int_algo, t, x0):
        dt = self.dt
        F = self.system.derive
        res = int_algo(F, t, x0, t + dt)
        res.step()
        dense_output = res.dense_output()
        if self.verbose:
            print(dense_output.t_min)
            print(dense_output.t_max)
        x1 = dense_output(t + dt)
        return x1

    def scipy_rk23(self, t, x0):
        import scipy.integrate
        int_algo = scipy.integrate.RK23
        return self._scipy_ode(int_algo, t, x0)

    def scipy_rk45(self, t, x0):
        import scipy.integrate
        int_algo = scipy.integrate.RK45
        return self._scipy_ode(int_algo, t, x0)

    def scipy_radau(self, t, x0):
        import scipy.integrate
        int_algo = scipy.integrate.Radau
        return self._scipy_ode(int_algo, t, x0)

    def scipy_bdf(self, t, x0):
        import scipy.integrate
        int_algo = scipy.integrate.BDF
        return self._scipy_ode(int_algo, t, x0)

    def scipy_lsoda(self, t, x0):
        import scipy.integrate
        int_algo = scipy.integrate.LSODA
        return self._scipy_ode(int_algo, t, x0)


def plot_states(states, **kwargs):
    import matplotlib.pyplot as plt
    external_data = kwargs.get('external_data', {})
    title = kwargs.get('title', 'State variables')
    fig, ax = plt.subplots()
    for n in states.dtype.names[1:]:
        ax.plot(states['t'], states[n], label=n)
    for k in external_data:
        if np.sum(np.abs(external_data[k][:, 1])) > 1e-10:
            ax.plot(external_data[k][:, 0], external_data[k][:, 1], label=k)
    ax.set(xlabel='time (s)', ylabel='states', title=title)
    ax.grid()
    ax.legend(loc='upper right')
    # fig.savefig("test.png")
    plt.show()
    return fig


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
    integration_algorithm = kwargs.get('integrator', 'rk44')
    verbose = kwargs.get('verbose', False)
    m = kwargs.get('m', 1)
    c = kwargs.get('c', 0)
    k = kwargs.get('k', 1)
    t_stop = kwargs.get('t_stop', 1000)
    dt = kwargs.get('dt', 0.1)
    x0 = kwargs.get('x0', 0.0)
    v0 = kwargs.get('v0', 0.0)
    initial_states = [x0, v0]
    instance = system(m=m, c=c, k=k)
    #
    t = np.arange(0, t_stop + dt, dt)
    d = 1 - np.arange(0, t_stop + dt, dt)/t_stop
    f = 0 * np.sin(2*np.pi*t/(t_stop/50)) * d
    tf = np.concatenate((np.vstack(t), np.vstack(f)),axis=1)

    integrator = Integrator(instance, external_data={'fext': tf}, verbose=verbose)
    states = integrator.integ(x0=initial_states, t_end=t_stop, dt=dt,
                              algorithm=integration_algorithm)
    #
    plot_states(states, external_data={'fext': tf}, title=str(instance))
    plot_f_wrt_x(states, external_data={'fext': tf})


def get_system_from_name(name):
    name_lower = name.lower()
    if name_lower == 'wb':
        system = WB
    elif name_lower == 'wb1':
        system = WB1
    elif name_lower == 'mck':
        system = MCK
    elif name_lower == 'vanderpol':
        system = VanDerPol
    else:
        raise Exception('System is not recognized')
    return system


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Simulate a mass/damper/spring system')
    pa = parser.add_argument
    pa('-s', '--system', help='Name of the system to simulate (default: will display a demo)',
                         default='')
    pa('-i', '--integrator',
       help='Integration algorithm. Available algorithms are {0}'.format(', '.join(Integrator.get_integration_algorithms())), default='rk44')
    pa('--dt', type=float, help='Integration time step (s)', default=0.1)
    pa('--tstop', type=float, help='Stop time (s)', default=10.0)
    pa('-m', type=float, help='Mass in kg', default=1.0)
    pa('-c', type=float, help='', default=0.0)
    pa('-k', type=float, help='', default=1.0)
    pa('--x0', type=float, help='Initial position (m)', default=0.0)
    pa('--v0', type=float, help='Initial speed (m/s)', default=0.0)
    pa('-v','--verbose', help='Display info', action='store_true')
    return parser


def main(cli=None):
    parser = get_parser()
    args = parser.parse_args(cli)
    if args.system:
        demo(system=get_system_from_name(args.system),
             integrator=args.integrator,
             t_stop=args.tstop,
             dt=args.dt,
             m=args.m,
             c=args.c,
             k=args.k,
             x0=args.x0,
             v0=args.v0,
             verbose=args.verbose)
    else:
        demo()


if __name__=='__main__':
    main()