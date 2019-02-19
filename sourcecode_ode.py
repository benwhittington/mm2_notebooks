import numpy as np
import matplotlib.pyplot as plt
import traitlets
from scipy.integrate import ode, odeint
import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.animation as animation

##########

# coursebook examples

##########

def ode0_ex_1_1_func(Tobj, t, Troom, alpha):
	dTdt = -1.*alpha*(Tobj-Troom)
	return dTdt

def ode0_ex_1_1(Tini,Troom,alpha,t):
	T = odeint(ode0_ex_1_1_func, Tini, t, args=(Troom, alpha))
	plt.xlabel("t")
	plt.ylabel("T(C)")
	plt.title("Newton's Law of Cooling")
	plt.plot(t,T)
	
def ode0_ex_1_2_func(theta, t, g, l):
	return [theta[1], -1.*(g/l)*np.sin(theta[0])]

def ode0_ex_1_2(theta1_ini,theta2_ini, t, g, l):
	theta0 = [theta1_ini, theta2_ini]
	theta = odeint(ode0_ex_1_2_func, theta0, t, args=(g,l))
	plt.xlabel("t")
	plt.ylabel("Angular displacement, (radians)")
	plt.title("Pendulum Anglular Displacement")
	plt.plot(t,theta[:,0])

##########

# convolution

##########

#  convolution functions

def shortSW(t):
    return [1*(val > 0 and val < 2) for val in t]

def longSW(t):
    return [1*(val > 0 and val < 4) for val in t]

def poly(t):
    return -0.25*(t)*(t-4)

def convolveStep(func1, func2, tao):
	""" 
		Compute one step of convolution

		----------

		Parameters

		----------

		func1: callable
			function to convolve

		func2: callable
			function to convolve

		tao: float
			dummy variable. Shift in t
	"""
	step = np.multiply(func1, func2)

	return np.trapz(step, x=tao)

def convoluteTo(f1, f2, t, tao):
	"""
		Convolve based on t values to tmax

		----------

		Parameters

		----------

		func1: callable
			function to convolve

		func2: callable
			function to convolve

		t: array_like
			time samples

		tao: float
			dummy variable. Shift in t

		----------

		Returns

		----------

		out: array_like
			convolved function
	"""
	out = []
	for i in t:
		out.append(np.trapz(np.multiply(f1(i-tao), f2(tao)), x=tao))

	return out

def convMain(tmax, func):
	"""
		Main function for producing convolution and plotting result

		----------

		Parameters

		----------

		tmax: float
			max t value

		func: string
			function key for selecting function
	"""
	_, (ax1, ax2) = plt.subplots(2, figsize=(16,8))

	f1 = shortSW
	tao = np.linspace(-10, 10, 1000)
	t = np.arange(-5, tmax, 0.5)

	if func=='f1': 
		func2 = shortSW(tmax-tao-0.5)
		f2 = shortSW

	elif func=='f2': 
		func2 = longSW(tmax-tao-0.5)
		f2 = longSW

	elif func=='f3':
		func2 = poly(tmax-tao-1.5)
		f2 = poly

	func1 = f1(tao)
	conv = convoluteTo(f1, f2, t, tao)

	ax1.plot(tao, func1)
	ax1.plot(tao, func2)
	ax1.set_xlim(-10, 10)
	ax1.set_ylim(0, 1.2)

	ax2.plot(t, conv)
	ax2.set_xlim(-10, 10)
	ax2.set_ylim(0, 4)

	ax1.set_xlabel('Tau')
	ax1.yaxis.set_ticks([])
	ax1.xaxis.set_ticks([])
	ax1.set_title('Functions showing t shift')
	ax2.set_xlabel('t')
	ax2.set_title('Convolution')
	ax2.yaxis.set_ticklabels([])
	ax2.xaxis.set_ticklabels([])


	plt.show()

def Convolution():
	"""
		main convolution funtion called from notebook
	"""
	t_sldr = widgets.FloatSlider(value=-5, min=-5, max=7, step=0.5, description='$t$', continuous_update=False)
	#  f3 doesn't really work properly
	# f_drp = widgets.Dropdown(options=['f1', 'f2', 'f3'])
	f_drp = widgets.Dropdown(options=['f1', 'f2'])

	return widgets.VBox([widgets.HBox([t_sldr, f_drp]), widgets.interactive_output(convMain, {'tmax':t_sldr, 'func':f_drp})])

##########

# numerical methods

##########

#  derivatives and solutions for solver
def dydx1(x, y, *args):
	return -y+1

def y1(x):
	return 1 - np.exp(-x)

def dydx2(x, y, *args):
	return x*np.exp(np.sin(x))

def dydx3(x, y, *args):
	return np.cos(x)

def y3(x):
	return np.sin(x)

def euler_step(f, xk, yk, h, pars=[]):
	''' Compute a single Euler step. From David Dempsey/ENGSCI 233
	
		Parameters
		----------
		f : callable
			Derivative function.
		xk : float
			Independent variable at beginning of step.
		yk : float
			Solution at beginning of step.
		h : float
			Step size.
		pars : iterable
			Optional parameters to pass to derivative function.
			
		Returns
		-------
		yk1 : float
			Solution at end of the Euler step.
	'''
	# evaluate derivative at point
	# step by h in direction of derivative
	# return value of new point
	return yk + h * f(xk, yk, *pars)
		
def euler_solve(f, x0, y0, x1, h, pars = []):
	''' Compute solution of initial value ODE problem using Euler method.
		From David Dempsey/ENGSCI 233

		----------
	
		Parameters

		----------

		f : callable
			Derivative function.

		x0 : float
			Initial value of independent variable.

		y0 : float
			Initial value of solution.

		x1 : float
			Final value of independent variable.

		h : float
			Step size.

		pars : iterable
			Optional parameters to pass into derivative function. 
			
		----------

		Returns

		----------

		xs : array-like
			Independent variable at solution.
		ys : array-like
			Solution.
			
	'''

	# initialise
	nx = int(np.ceil((x1-x0)/h))		# compute number of Euler steps to take
	xs = x0+np.arange(nx+1)*h			# x array
	ys = 0.*xs							# array to store solution
	ys[0] = y0							# set initial value
	
	for i in range(len(xs)-1):
		ys[i+1] = euler_step(f, xs[i], ys[i], h, pars)

	return xs, ys

def improved_euler_step(f, xk, yk, h):
	""" 
		Completes one step of the improved euler method

		----------

		Parameters

		----------

		f: callable
			differential equation to solve

		xk: float
			x ordinate for step start

		tk: float
			y ordinate for step start
		
		h: float
			step size

		----------

		Returns

		----------
		float: Approximated yk+1 value

	"""

	return yk + h * 0.5*(f(xk, yk) + f(xk + h, yk + h * f(xk, yk)))

def improved_euler_solve(f, x0, y0, x1, h):
	""" 
		solves a differential equation using improved euler

		----------

		Parameters

		----------

		f: callable
			differential equation to solve	

		x0: float 
			inital x value

		y0: float 
			inital y value

		x1: float
			upper x value

		h: float
			step size

		----------

		Returns

		----------
		xs: array_like
			x time steps used

		ys: approximated y values
		
	"""
	# initialise
	nx = int(np.ceil((x1-x0)/h))		# compute number of Euler steps to take
	xs = x0+np.arange(nx+1)*h			# x array
	ys = 0.*xs							# array to store solution
	ys[0] = y0							# set initial value

	for i in range(len(xs)-1):
		# solve for next value in function
		ys[i+1] = improved_euler_step(f, xs[i], ys[i], h)

	return xs, ys

def runge_step(f, xk, yk, h, *args):
	""" 
		completes a single step of the runge kutta method (K4)

		----------

		Parameters

		----------
		
		f: callable
			differential equation to solve

		xk: float
			x ordinate for step start

		tk: float
			y ordinate for step start
		
		h: float
			step size

		----------

		Returns

		----------

		float: Approximated yk+1 value
	"""
	k1 = h*f(xk, yk, *args)
	k2 = h*f(xk + h/2, yk + k1/2, *args)
	k3 = h*f(xk + h/2, yk + k2/2, *args)
	k4 = h*f(xk + h, yk + k3, *args)

	return yk + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

def runge_solve(f, x0, y0, x1, h, *args):
	''' 
		solves a differential equation using runge kutta

		----------

		Parameters

		----------
		
		f: callable
			differential equation to solve
		
		x0: float
			x inital condition

		y0: float
			y initial condition
		
		x1: float
			x stopping point
		
		h: float
			step size

		----------

		Returns

		----------

		xs: array_like
			x time steps used

		ys: approximated y values
	'''
	nx = int(np.ceil((x1-x0)/h))
	xs = x0+np.arange(nx+1)*h
	ys = 0.*xs
	ys[0] = y0		

	for i in range(len(xs)-1):
		ys[i+1] = runge_step(f, xs[i], ys[i], h, args)

	return xs, ys

def plotTangentAtPoint(ax, f, x0, y0, xMin, xMax, h, label=None, color='b'):
	''' plots line on axis given function derivative and x/y points

		----------

		Parameters

		----------

		ax: matplotlib object
			axis to plot on
		f: callable
			derivative function
		x: float
			x ordinate of point
		y: float
			y ordinate of point

	'''
	# plots in range 0 < x < 10

	x = np.arange(xMin,xMax,1)
	m = f(x0,y0)
	y = m*(x-x0) + y0
	if label is not None:
		ax.plot(x,y, ls='--', linewidth=1, label=label, color=color)
	else:
		ax.plot(x,y, ls='--', linewidth=1, color=color)
	
def RunMethod(h, steps, eqnKey, method, showSln, showGrad):
	''' Runs demo of particular numerical method show

		----------

		Parameters

		----------

        h: float
			step size

        steps: int
			no. steps to take

        eqnKey: str
			key for dict of equtions to use

        method: str
			string key for which method to use

        showSln: bool
			bool for whether to show analytic sln

        showGrad: bool
			bool for whether to show ext grad evals
	'''

	xMin = 0
	xMax = 20
	yMin = 0

	_ = plt.figure(figsize=(16,8))
	ax = plt.axes()
	# get equation
	equation = {"dydx = -y +1":[dydx1, y1, 0, 20, 0, 1.2], "dy/dx = cos(x)":[dydx3, y3, 0, 20, -1.5, 1.5]}
	f, soln, xMin, xMax, yMin, yMax = equation[eqnKey]

	xStop = h * steps
	if xStop > xMax: xStop = xMax
    
	if method=='Improved Euler': # call to improved euler
		x, y = improved_euler_solve(f, 0, 0, xStop, h)
		# for displaying next gradient evaluations
		if showGrad:
			plotTangentAtPoint(ax,f,x[-1]+h,euler_step(f, x[-1], y[-1], h), x[-1]+h, xMax,h, label='Corrector Step', color='r')
			plotTangentAtPoint(ax,f,x[-1],y[-1], x[-1], xMax,h, label='Predictor Step', color='b')
	elif method=='Euler':# call to euler
		x, y = euler_solve(f, 0, 0, xStop, h)
		# for displaying next gradient evaluations
		if showGrad:
			plotTangentAtPoint(ax,f,x[-1],y[-1], xMin, xMax,h, label='Gradient Evaluation')
	elif method=='Runge Kutta': # call to runge Kutta
		x, y = runge_solve(f, 0, 0, xStop, h)

	# plot numerical
	ax.plot(x, y, marker='o',color='k', label='Numerical Solution')

	# generate values for error
	xPlot = np.arange(0, 20, 0.1)
	ySoln = soln(xPlot)
	error = sum((soln(x)[:len(y)]-y)**2)
	# add text showing error
	ax.text(0.01,0.95, 'Total sum of squares error = {:.2e}'.format(error), transform=ax.transAxes, size=20)

	# plot analytic
	if showSln: ax.plot(xPlot,ySoln, label='Analytic Solution')

	plt.xlim(xMin, xMax)
	plt.ylim(yMin, yMax)
	ax.legend(loc=1)
	plt.show()

def NumericalMethods():
	"""
		Main function for called by notebook to produce plots and UI
	"""
	steps_sldr = widgets.IntSlider(value=3, description='Steps', min=0, max=50, continuos_update=False)
	h_sldr = widgets.FloatSlider(value=0.5, description='Step Size', min=0.1, max=1, step=0.1,)
	solver_drp = widgets.Dropdown(options=['Euler', 'Improved Euler', 'Runge Kutta'], description='Solver Type')
	eqn_drp = widgets.Dropdown(options=["dydx = -y +1", "dy/dx = cos(x)"], description='Derivative')
	showSln_chk = widgets.Checkbox(value=False, description='Show Solution')
	grad_chk = widgets.Checkbox(value=False, description='Show Next Gradient Evaluations')

	return widgets.VBox([
		widgets.VBox([
			widgets.HBox([
				h_sldr, 
				eqn_drp, 
				grad_chk]), 
			widgets.HBox([
				steps_sldr, 
				solver_drp, 
				showSln_chk])]), 
		widgets.interactive_output(RunMethod, {
			'h':h_sldr, 
			'steps':steps_sldr, 
			'method':solver_drp, 
			'eqnKey':eqn_drp,
			'showSln':showSln_chk,
			'showGrad':grad_chk})
			])

##########

# laplace transforms

# t* = function in t domain
# s* function in s domain

# probably a much better way of doing this. Maybe function class with laplace and t attributes?

##########

def tPoly(t, n, tShift, sShift):
	full = (t-tShift)**n*np.exp(sShift*t)
	return [0 if ((t[i] < 0) or (t[i] < tShift)) else full[i] for i in range(len(t))]

def sPoly(s, n, tShift, sShift):
	full = np.exp(-tShift*s)*np.math.factorial(n)/(s-sShift)**(n+1)
	return [0 if ((s[i] < 0) or (s[i] < sShift)) else full[i] for i in range(len(s))]
	
def tSin(t, w, tShift, sShift):
	full = np.sin(w*(t-tShift))*np.exp(sShift*t)
	return [0 if ((t[i] < 0) or (t[i] < tShift)) else full[i] for i in range(len(t))]

def sSin(s, w, tShift, sShift):
	full = np.exp(-tShift*s)*(w/((s-sShift)**2 + w**2))
	return [0 if ((s[i] < 0) or (s[i] < sShift)) else full[i] for i in range(len(s))]

def tCos(t, w, tShift, sShift):
	full = np.cos(w*(t-tShift))*np.exp(sShift*t)
	return [0 if ((t[i] < 0) or (t[i] < tShift)) else full[i] for i in range(len(t))]

def sCos(s, w, tShift, sShift):
	full = np.exp(-tShift*s)*(s-sShift)/((s-sShift)**2 + w**2)
	return [0 if ((s[i] < 0) or (s[i] < sShift)) else full[i] for i in range(len(s))]

def tExp(t, a, tShift, sShift):
	full = np.exp(a*(t-tShift))*np.exp(sShift*t)
	return [0 if ((t[i] < 0) or (t[i] < tShift)) else full[i] for i in range(len(t))]

def sExp(s, a, tShift, sShift):
	full = np.exp(-tShift*s)/((s-sShift)-a)
	return [0 if ((s[i] < 0) or (s[i] < sShift)) else full[i] for i in range(len(s))]
	
def Laplace():
	"""
		Main function called by notebook to produce plots and UI
	"""

	# for ax lim sliders
	tAbsLim = 20
	sAbsLim = 20

	# define widgets
	tlim_sldr = widgets.FloatRangeSlider(value=[-10, 10], step=0.1, min=-tAbsLim, max=tAbsLim, description='t lim', continuous_update=False)
	slim_sldr = widgets.FloatRangeSlider(value=[-10, 10], step=0.1, min=-sAbsLim, max=sAbsLim, description='s lim', continuous_update=False)
	fslim_sldr = widgets.FloatRangeSlider(value=[0, 5], step=0.1, min=-5, max=5, description='F(s) lim', continuous_update=False)
	tShift_sldr = widgets.FloatSlider(value=0, step=0.1, min=-2, max=2, description='Shift t', continuous_updating=False)
	sShift_sldr = widgets.FloatSlider(value=0, step=0.1, min=-2, max=2, description='Shift s', continuous_updating=False)
	func_drop = widgets.Dropdown(options={'t^n':'Poly', 'sin(w*t)':'Sine', 'cos(w*t)':'Cos', 'exp(at)':'exp'}, layout=widgets.Layout(width='220px'), description='$f(t)=$')

	# display
	display(widgets.VBox([
		widgets.HBox([
			widgets.VBox([
				tlim_sldr,
				slim_sldr
			]),
			widgets.VBox([
				tShift_sldr,
				sShift_sldr
			]), 
			widgets.VBox([
				fslim_sldr,
				func_drop
			])
		]),
		widgets.interactive_output(showLaplace, {
			'slim':slim_sldr,
			'tlim':tlim_sldr,
			'func':func_drop,
			'tShift':tShift_sldr,
			'fslim':fslim_sldr,
			'sShift':sShift_sldr
			})
		]))

def computeTransform(t, s, slim, tlim, var, tfunc, sfunc, tShift, sShift, fslim):
	"""
		Provides function for interactive output of n or omega to work

		----------

		Parameters

		----------

		t: array_like
			time values

		s: array_like
			s values

		slim: array_like
			pair of s limit values
		
		tlim: array_like
			pair of t limit values

		var: float
			variable storing either omega or n

		tfunc: callable
			t domain function

		sfunc: callable
			s domain function

		tShift: float
			shift in t

		sShift: float
			 shift in s

		fslim: array_like
			s domain y lim pair
	"""

	ft = tfunc(t, var, tShift, sShift)
	fs = sfunc(s, var, tShift, sShift)

	plotLaplace(t, s, ft, fs, tlim, slim, fslim)

def showLaplace(slim, tlim, func, tShift, sShift, fslim):
	"""
		Chooses function and transform to plot. Produces t samples

		----------

		Parameters

		----------

		slim: array_like
			pair of points for s limits

		tlim: array_like
			pair of points for t limits

		func: str
			string for selected function

		tShift: float
			shift in t

		sShift: float
			 shift in s

		fslim: array_like
			s domain y lim pair
	"""
	t = np.arange(-20, 20, 0.01)
	# s = np.arange(-50, 50 1)
	s = t.copy()
	if func=='Poly':
		var_sldr=widgets.IntSlider(1, min=0, max=10, step=1, description='$n$', continuous_update=False)
		tfunc = tPoly
		sfunc = sPoly

	elif func=='Sine':
		var_sldr=widgets.FloatSlider(1, min=0.1, max=10, step=0.1, description='$\\omega$', continuous_update=False)
		tfunc = tSin
		sfunc = sSin

	elif func=='Cos':
		var_sldr=widgets.FloatSlider(1, min=0.1, max=10, step=0.1, description='$\\omega$', continuous_update=False)
		tfunc = tCos
		sfunc = sCos		

	elif func=='exp':
		var_sldr=widgets.FloatSlider(1, min=-2, max=2, step=0.1, description='$a$', continuous_update=False)
		tfunc = tExp
		sfunc = sExp

	display(widgets.VBox([var_sldr, widgets.interactive_output(computeTransform, {
		't':widgets.fixed(t),
		's':widgets.fixed(s),
		'slim':widgets.fixed(slim), 
		'tlim':widgets.fixed(tlim), 
		'var':var_sldr,
		'tfunc':widgets.fixed(tfunc),
		'sfunc':widgets.fixed(sfunc),
		'tShift':widgets.fixed(tShift),
		'sShift':widgets.fixed(sShift),
		'fslim':widgets.fixed(fslim)
		})]))

def plotLaplace(t, s, ft, fs, tlim, slim, fslim):
	"""
		Main plotting function for laplace demo

		----------

		Parameters

		----------

		t: array_like
			time samples

		s: array_like
			s samples

		ft: array_like
			points in t space

		fs: array_like
			points in s space

		tlim: array_like
			pair of points for t limits

		slim: array_like
			pair of points for s limits

		fslim: array_like
			s domain y lim
	"""

	_, (tAx, sAx) = plt.subplots(1, 2, figsize=(16,8))

	tAx.plot(t, ft)
	tAx.set_xlim(*tlim)
	tAx.set_title('t Domain', fontsize=20)
	tAx.set_xlabel('t', fontsize=16)
	tAx.set_ylabel('f(t)', fontsize=16)

	sAx.plot(s, fs)
	sAx.set_title('s Domain', fontsize=20)
	sAx.set_xlabel('s', fontsize=16)
	sAx.set_xlim(*slim)
	sAx.set_ylim(*fslim)
	sAx.set_ylabel('F(s)', fontsize=16)
	
	plt.show()

##########

# ode_suspension

##########

def Suspension():
	"""
		Suspension function to be called from notebook.
	"""
	c_sldr = widgets.FloatSlider(3, min=0, max=3, step=0.1, description='c', continuous_update=False)
	m_sldr = widgets.FloatSlider(1, min=0.1, max=3, step=0.1, description='m', continuous_update=False)
	k_sldr = widgets.FloatSlider(1, min=0.1, max=3, step=0.1, description='k', continuous_update=False)

	return (
		widgets.VBox([
			widgets.HBox([
				c_sldr,
				m_sldr,
				k_sldr
			]),
			widgets.interactive_output(suspensionMain, {'c':c_sldr, 'm':m_sldr, 'k':k_sldr})
			])
	)

def suspensionDE(Y, t, m, c, k):
	"""
		Colin Simpson's code...
		But the beautiful docstring is all mine

		Y: array_like
			contains initial conditions
		
		t: array_like
			time values

		m: float
			mass parameter

		c: float
			damping coeffcient parameter
		
		k: float
			spring constant parameter
	"""

	return [Y[1], -c*Y[1]-k*Y[0]/m]

def suspensionMain(c, m, k):

	"""
		Built on code from Colin Simpson.
		Main code for computing suspension behaviour and returning values for plotting.

		m: float
			mass parameter

		c: float
			damping coeffcient parameter
		
		k: float
			spring constant parameter
	"""

	tmin = 0
	tmax = 15
	t = np.arange(tmin, tmax, 0.1)

	# start at y = 1, dydt = 0
	init = [1, 0]
	
	y = odeint(suspensionDE, init , t, args=(m, c, k))

	if c*c > 4*m*k:
		title = 'y vs t: over-damping'
	elif c*c == 4*m*k:
		title = 'y vs t: critical-damping'
	elif c == 0:
		title = 'y vs t: no-damping'
	else: 
		title = 'y vs t: under-damping'
		
	plotSuspension(y, t, title, tmin, tmax)
    
def plotSuspension(y, t, title, tmin, tmax):
	"""
		Main function for plotting suspension output

		y: array_like
			vertical displacement
		
		t: array_like
			time values
		
		title: string
			graph title

		tmin: float
			minimum time value

		tmax: float maximum time value
	"""

	_ ,ax = plt.subplots(1,1, figsize=(16,8))
	plt.xlabel('t')
	plt.ylabel('y')
	ax.plot(t,y[:,0], color='C0')
	ax.set_title(title, fontsize=20)
	ax.set_ylim(-1.2, 1.2)
	ax.set_xlim(tmin, tmax)

	#ax.plot([t_i],y[1,0],marker='o', markersize=3, color="red")
	plt.show() 

##########

# pendulum

##########

def dvdt(a, t, *args):
	"""
		1st equation of system of ODEs

		----------

		Parameters

		----------

		a: float
			angle

		t: float (unused)
			time

		args[0]: g
			acceleration due to gravity

		args[1]: l 
			length of pendulum		

		----------

		Returns

		----------

		First DE od ode system
	"""
	g, l = args
	return (g/l)*np.sin(a)

def dadt(v, t, *args):
	"""
		1st equation of system of ODEs

		----------

		Parameters

		----------

		a: float
			angle

		t: float (unused)
			time

		----------

		Returns

		----------

		Second DE of ode system
	"""
	return v


def pendUpdate(i, aVec, vVec, line):
	"""
		Update function for animation

		----------

		Parameters

		----------

		i: int
			animation frame number

		aVec: array_like
			a single value. Don't ask why it's an array not just a float

		vVec: array_like
			v

		line: matplotlib object

		----------

		Returns

		----------

		line: matplotlib object
			line updated with new points

	"""
	h = 0.05
	vVec[0] = runge_step(dvdt, aVec[0], vVec[0], h, 9.81, 1)
	aVec[0] = runge_step(dadt, vVec[0], aVec[0], h)
	# print(vVec)
	line.set_data([0, np.sin(aVec)], [0, np.cos(aVec)])
	return line,

def Pendulum(a0=0, v0=0):
	"""
		Main function for calling matplotlib animation

		a0: float
			starting angle (rad)
		
		v0: float
			starting angular velocity
	"""
	aVec = [a0]
	vVec = [v0]

	fig, ax = plt.subplots(figsize=(12,12))

	line, = ax.plot([0, np.sin(aVec[0])], [0, np.cos(aVec[0])])

	ax.set_xlim(-1.2,1.2)
	ax.set_ylim(-1.2,1.2)
	# frames = 1000 slow to load but good play speed for notebook
	# interval = 5
	anim = animation.FuncAnimation(fig, pendUpdate, frames=150, interval=50, fargs=[aVec, vVec, line], repeat=True, blit=True)
	
	if __name__=="__main__":
		plt.show()
	else:
		return HTML(anim.to_jshtml())

def pendMain():
	"""
		Main pendulum function called by notebook
	"""
	methods = {'Runge':runge_step, 'Euler':euler_step, 'Improved':improved_euler_step}

	method_drop = widgets.Dropdown(options=methods)

	display(
		method_drop,
		widgets.interactive_output(Pendulum, {})
	)

if __name__=="__main__":
	# pendMain()
	Pendulum()


