import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import ipywidgets as wid
import traitlets
from ipywidgets import interact, fixed
import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import integrate
from scipy.optimize import root
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def mc1_example_2_1(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = t
	ry = t

	# plot object motion as dots/line
	fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_xlim([-0.1,4.1])
	ax1.set_ylim([-0.1,4.1])
	ax1.plot(rx,ry,'k-o')
	
	
def mc1_example_2_2(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = -2+2*np.cos(t)
	ry = 2+2*np.sin(t)

	# plot object motion as dots/line
	fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_xlim([-4.1,0.1])
	ax1.set_ylim([-0.1,4.1])
	ax1.plot(rx,ry,'k-o')

def mc1_example_2_3(ts,te,tstep):	
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = np.cos(t)
	ry = np.sin(t)
	rz = t

	# plot object motion as dots/line
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.set_xlim([-1.1,1.1])
	ax1.set_ylim([-1.1,1.1])
	ax1.set_zlim([-0.1,2.*np.pi+0.1])
	ax1.plot(rx,ry,rz,'k-o')
	
	
def mc1_example_2_4(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = t*0-2
	ry = 5*np.cos(t)
	rz = np.sin(t)

	# plot object motion as dots/line
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.set_xlim([-2.1,-1.9])
	ax1.set_ylim([-5.1,5.1])
	ax1.set_zlim([-1.1,1.1])
	ax1.plot(rx,ry,rz,'k-o')

	ax2 = fig.add_subplot(122)
	ax2.set_xlabel('y(t)')
	ax2.set_ylabel('z(t)')
	ax2.set_xlim([-5.1,5.1])
	ax2.set_ylim([-1.1,1.1])
	ax2.plot(ry,rz,'k-o',label='x=-2')
	ax2.legend()
	
def mc1_example_2_5(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = t+np.sin(t)
	ry = 1+np.cos(t)

	# plot object motion as dots/line
	fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_xlim([-0.1,9.1])
	ax1.set_ylim([-0.1,2.1])
	ax1.plot(rx,ry,'k-o')

def mc1_example_3_1(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = np.cos(t)
	ry = np.sin(t)
	rz = t
	vx = -1.*np.sin(t)
	vy = np.cos(t)
	vz = 1.

	# plot object trajectory
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.set_xlim([-1.1,1.1])
	ax1.set_ylim([-1.1,1.1])
	ax1.set_zlim([-0.1,2*np.pi+0.1])
	ax1.plot(rx,ry,rz,'k-o',label='r(t)')
	ax1.legend()

	# plot object velocity
	ax2 = fig.add_subplot(122, projection='3d')
	ax2.set_xlabel('dx/dt')
	ax2.set_ylabel('dy/dt')
	ax2.set_zlabel('dz/dt')
	ax2.set_xlim([-1.1,1.1])
	ax2.set_ylim([-1.1,1.1])
	ax2.set_zlim([0.9,1.1])
	ax2.plot(vx,vy,vz,'b-o',label='v(t)')
	ax2.legend()

def mc1_example_4_1(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = 2.*np.cos(t)
	ry = 2.*np.sin(t)
	rz = 0*t
	dist = 2.*t

	# plot object trajectory
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.set_xlim([-2.1,2.1])
	ax1.set_ylim([-2.1,2.1])
	ax1.set_zlim([-0.1,0.1])
	ax1.plot(rx,ry,rz,'k-o',label='r(t)')
	ax1.legend()

	# plot object distance travelled
	ax2 = fig.add_subplot(122)
	ax2.set_xlabel('t')
	ax2.set_ylabel('s')
	ax2.set_xlim([-0.1,np.pi+0.1])
	ax2.set_ylim([-0.1,2.*np.pi+0.1])
	ax2.plot(t,dist,'k-o',label='s(t)')
	ax2.legend()

def mc1_example_4_2(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = 2.*t
	ry = 3.*np.sin(2.*t)
	rz = 3.*np.cos(2.*t)
	dist = 2.*np.sqrt(10.)*t

	# plot object trajectory
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	#ax1.set_xlim([-2.1,2.1])
	#ax1.set_ylim([-2.1,2.1])
	#ax1.set_zlim([-0.1,0.1])
	ax1.plot(rx,ry,rz,'k-o',label='r(t)')
	ax1.legend()

	# plot object distance travelled
	ax2 = fig.add_subplot(122)
	ax2.set_xlabel('t')
	ax2.set_ylabel('s')
	#ax2.set_xlim([-0.1,np.pi+0.1])
	#ax2.set_ylim([-0.1,2.*np.pi+0.1])
	ax2.plot(t,dist,'k-o',label='s(t)')
	ax2.legend()

def mc1_example_arclength(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = np.sin(t)
	ry = np.cos(t)
	dist = t

	# plot object trajectory
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(121)
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_xlim([-0.1,1.1])
	ax1.set_ylim([-1.1,1.1])
	ax1.plot(rx,ry,'k-o',label='r(t)')
	ax1.legend()

	# plot object distance travelled
	ax2 = fig.add_subplot(122)
	ax2.set_xlabel('t')
	ax2.set_ylabel('s')
	ax2.set_xlim([-0.1,np.pi+0.1])
	ax2.set_ylim([-0.1,np.pi+0.1])
	ax2.plot(t,dist,'k-o',label='s(t)')
	ax2.legend()

def mc1_example_5_1(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = np.cos(t)*np.cos(t)
	ry = -2.*np.sin(2.*t)
	rz = t*t
	
	# tangent vector i.e. velocity
	vx = -2.*np.cos(t)*np.sin(t)
	vy = -4.*np.cos(2.*t)
	vz = 2.*t
	
	# unit tangent vector
	norm = 1./(vx*vx+vy*vy+vz*vz)
	utvx = norm*vx
	utvy = norm*vy
	utvz = norm*vz
	
	# plot object trajectory
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.set_xlim([-0.1,1.1])
	ax1.set_ylim([-2.1,0.1])
	ax1.set_zlim([-0.1,np.pi*np.pi+0.1])
	ax1.quiver(rx[-1],ry[-1],rz[-1],utvx[-1],utvy[-1],utvz[-1], pivot='middle',normalize=True, label='unit tangent vector')
	ax1.plot(rx,ry,rz,'ko',label='r(t)')
	ax1.legend()
	
def mc1_example_7_1(ts,te,tstep):
	# set time and position vector
	t = np.arange(ts,te+tstep,tstep)
	rx = t
	ry = 2.*t

	# set force vector field
	X, Y = np.meshgrid(np.arange(0,1,0.1), np.arange(0,2,0.1))
	U = Y*Y
	V = -1*(X*X)

	
	# plot force field as arrows and object motion as blue dots/line
	fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
	ax1.quiver(X, Y, U, V, units='width')
	ax1.plot(rx,ry,'b-o')
	
def mc1_example_7_3():
	# set force vector field
	X, Y = np.meshgrid(np.arange(-1,1,0.1), np.arange(-1,1,0.1))
	U = -1.*Y
	V = X*Y

	# set time and position vector
	t = np.linspace(0,np.pi/2.,11)
	rx = np.cos(t)
	ry = np.sin(t)

	# plot force field as arrows and object motion as blue dots/line
	fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
	ax1.quiver(X, Y, U, V, units='width')
	ax1.plot(rx,ry,'b-o')

def mc1_example_7_4():
	# set force vector field
	X, Y, Z = np.meshgrid(np.arange(-0.2,1.2,0.2),
						  np.arange(-0.2,1.2,0.2),
						  np.arange(-0.2,1.2,0.2))
	U = X
	V = -1*Z
	W = 2*Y

	# set time and position vector for each segment
	t = np.linspace(0,1,11)
	rx1 = t
	ry1 = t
	rz1 = t*0
	rx2 = t*0+1
	ry2 = t*0+1
	rz2 = t
	rx3 = 1-t
	ry3 = 1-t
	rz3 = 1-t

	# plot object motion as dots/line
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.set_xlabel('x(t)')
	ax1.set_ylabel('y(t)')
	ax1.set_zlabel('z(t)')
	ax1.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)
	ax1.plot(rx1,ry1,rz1,'b-o')
	ax1.plot(rx2,ry2,rz2,'b-o')
	ax1.plot(rx3,ry3,rz3,'b-o')
	
	
	
	
def mc3_example_2_1_func(x,y):
    return (x*x*y)

def mc3_example_2_1(option):
	nxy = 51
	x = np.linspace(0.,1.,nxy)
	y = np.linspace(3.,4.,nxy)
	X,Y = np.meshgrid(x, y)
	Z = mc3_example_2_1_func(X, Y)
	
	verts1 = []
	verts2 = []
	verts3 = []
	if option == 'x':
		verts1.append([x[np.int(nxy/2)],y[0],0.])
		for i in range(nxy):
			verts1.append([x[np.int(nxy/2)],y[i],mc3_example_2_1_func(x[np.int(nxy/2)], y[i])])
		verts1.append([x[np.int(nxy/2)],y[-1],0.])
		verts2.append([x[np.int(nxy/4)],y[0],0.])
		for i in range(nxy):
			verts2.append([x[np.int(nxy/4)],y[i],mc3_example_2_1_func(x[np.int(nxy/4)], y[i])])
		verts2.append([x[np.int(nxy/4)],y[-1],0.])
		verts3.append([x[np.int(nxy/1.2)],y[0],0.])
		for i in range(nxy):
			verts3.append([x[np.int(nxy/1.2)],y[i],mc3_example_2_1_func(x[np.int(nxy/1.2)], y[i])])
		verts3.append([x[np.int(nxy/1.2)],y[-1],0.])
	elif option == 'y':
		verts1 = []
		verts1.append([x[0],y[np.int(nxy/2)],0.])
		for i in range(nxy):
			verts1.append([x[i],y[np.int(nxy/2)],mc3_example_2_1_func(x[i],y[np.int(nxy/2)])])
		verts1.append([x[-1],y[np.int(nxy/2)],0.])
		verts2.append([x[0],y[np.int(nxy/4)],0.])
		for i in range(nxy):
			verts2.append([x[i],y[np.int(nxy/4)],mc3_example_2_1_func(x[i],y[np.int(nxy/4)])])
		verts2.append([x[-1],y[np.int(nxy/4)],0.])
		verts3.append([x[0],y[np.int(nxy/1.2)],0.])
		for i in range(nxy):
			verts3.append([x[i],y[np.int(nxy/1.2)],mc3_example_2_1_func(x[i],y[np.int(nxy/1.2)])])
		verts3.append([x[-1],y[np.int(nxy/1.2)],0.])
	
	fig = plt.figure(figsize=(16, 8))
	if option == 'default':
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		ax1.set_zlim([0,4])
		ax1.set_title('Surface')
		ax1.plot_surface(X, Y, 0*Z, rstride=1, cstride=1, color='0.75', linewidth=0, antialiased=True)
		ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
		ax1.view_init(30, 315)
		ax2 = fig.add_subplot(122)
		ax2.set_xlabel('x')
		ax2.set_ylabel('y')
		ax2.set_title('Integration Region')
		ax2.plot([x[0],x[-1],x[-1],x[0],x[0]],[y[0],y[0],y[-1],y[-1],y[0]], 'k--o')
	else:
		ax1 = fig.add_subplot(111, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		ax1.set_zlim([0,4])
		ax1.plot_wireframe(X, Y, 0*Z, rstride=nxy, cstride=nxy, color='k', linewidth=1.0, antialiased=True)
		ax1.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=0.4)
		face1 = Poly3DCollection([verts1], linewidth=1, alpha=0.5)
		face2 = Poly3DCollection([verts2], linewidth=1, alpha=0.5)
		face3 = Poly3DCollection([verts3], linewidth=1, alpha=0.5)
		face1.set_facecolor((0, 0, 1, 0.5))
		face2.set_facecolor((0, 0, 1, 0.5))
		face3.set_facecolor((0, 0, 1, 0.5))
		ax1.add_collection3d(face1)
		ax1.add_collection3d(face2)
		ax1.add_collection3d(face3)
		ax1.view_init(30, 120)
		
def mc3_example_3_1_func(x,y):
    return (y)

def mc3_example_3_1(option):
	nxy = 101
	x = np.linspace(0.,2.,nxy)
	y = 2.*x
	X,Y = np.meshgrid(x, y)
	Z = mc3_example_3_1_func(X, Y)

	verts = []
	verts.append([x[-1],y[0],0.])
	for i in range(nxy):
		verts.append([x[i],y[i],0.])
	verts.append([x[-1],y[0],0.])

	if option != 'default':
		verts1 = []
		verts2 = []
		verts3 = []
		
		if option == 'x':
			tmp = np.int(nxy/2)
			verts1.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts1.append([x[tmp],y[i],mc3_example_3_1_func(x[tmp],y[i])])
			verts1.append([x[tmp],y[tmp],0.])

			tmp = np.int(nxy/4)
			verts2.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts2.append([x[tmp],y[i],mc3_example_3_1_func(x[tmp],y[i])])
			verts2.append([x[tmp],y[tmp],0.])

			tmp = np.int(nxy/1.2)
			verts3.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts3.append([x[tmp],y[i],mc3_example_3_1_func(x[tmp],y[i])])
			verts3.append([x[tmp],y[tmp],0.])	

		if option == 'y':
			tmp = np.int(nxy/2)
			verts1.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts1.append([x[tmp+i],y[tmp],mc3_example_3_1_func(x[tmp+i],y[tmp])])
			verts1.append([x[-1],y[tmp],0.])

			tmp = np.int(nxy/4)
			verts2.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts2.append([x[tmp+i],y[tmp],mc3_example_3_1_func(x[tmp+i],y[tmp])])
			verts2.append([x[-1],y[tmp],0.])

			tmp = np.int(nxy/1.2)
			verts3.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts3.append([x[tmp+i],y[tmp],mc3_example_3_1_func(x[tmp+i],y[tmp])])
			verts3.append([x[-1],y[tmp],0.])
		
	fig = plt.figure(figsize=(16, 8))
	if option == 'default':
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		ax1.set_zlim([0,4])
		ax1.set_title('Surface')
		ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
		ax1.view_init(30, 325)
		ax2 = fig.add_subplot(122)
		ax2.set_xlabel('x')
		ax2.set_ylabel('y')
		ax2.set_title('Integration region')
		ax2.plot([x[0],x[-1]],[y[0],y[0]], 'k--o')
		ax2.plot([x[-1],x[-1]],[y[0],y[-1]], 'k--o')
		ax2.plot([x[-1],x[0]],[y[-1],y[0]], 'k--o')
		r = Poly3DCollection([verts], linewidth=1, alpha=0.5)
		r.set_facecolor((0.2, 0.2, 0.2, 0.5))
		ax1.add_collection3d(r)
	else:
		ax1 = fig.add_subplot(111, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		ax1.set_zlim([0,4])
		#ax1.plot_wireframe(X, Y, 0*Z, rstride=nxy, cstride=nxy, color='k', linewidth=1.0, antialiased=True)
		ax1.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=0.4)
		face1 = Poly3DCollection([verts1], linewidth=1, alpha=0.5)
		face2 = Poly3DCollection([verts2], linewidth=1, alpha=0.5)
		face3 = Poly3DCollection([verts3], linewidth=1, alpha=0.5)
		face1.set_facecolor((0, 0, 1, 0.5))
		face2.set_facecolor((0, 0, 1, 0.5))
		face3.set_facecolor((0, 0, 1, 0.5))
		ax1.add_collection3d(face1)
		ax1.add_collection3d(face2)
		ax1.add_collection3d(face3)
		ax1.view_init(30, 315)
		r = Poly3DCollection([verts], linewidth=1, alpha=0.5)
		r.set_facecolor((0.2, 0.2, 0.2, 0.5))
		ax1.add_collection3d(r)


def mc3_example_3_2_func(x,y):
    return (np.exp(x*x*x))

def mc3_example_3_2(option):
	nxy = 101
	y = np.linspace(0.,4.,nxy)
	x = np.sqrt(y)
	X,Y = np.meshgrid(x, y)
	Z = mc3_example_3_2_func(X, Y)

	intreg_x = []
	intreg_y = []
	for i in range(nxy):
		intreg_x.append(x[i])
		intreg_y.append(y[i])
	intreg_x.append(x[-1])
	intreg_y.append(y[0])
	intreg_x.append(x[0])
	intreg_y.append(y[0])

	if option != 'default':
		verts1 = []
		verts2 = []
		verts3 = []
		
		if option == 'x':
			tmp = np.int(nxy/2)
			verts1.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts1.append([x[tmp],y[i],mc3_example_3_2_func(x[tmp],y[i])])
			verts1.append([x[tmp],y[tmp],0.])

			tmp = np.int(nxy/4)
			verts2.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts2.append([x[tmp],y[i],mc3_example_3_2_func(x[tmp],y[i])])
			verts2.append([x[tmp],y[tmp],0.])

			tmp = np.int(nxy/1.2)
			verts3.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts3.append([x[tmp],y[i],mc3_example_3_2_func(x[tmp],y[i])])
			verts3.append([x[tmp],y[tmp],0.])	

		if option == 'y':
			tmp = np.int(nxy/2)
			verts1.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts1.append([x[tmp+i],y[tmp],mc3_example_3_2_func(x[tmp+i],y[tmp])])
			verts1.append([x[-1],y[tmp],0.])

			tmp = np.int(nxy/4)
			verts2.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts2.append([x[tmp+i],y[tmp],mc3_example_3_2_func(x[tmp+i],y[tmp])])
			verts2.append([x[-1],y[tmp],0.])

			tmp = np.int(nxy/1.2)
			verts3.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts3.append([x[tmp+i],y[tmp],mc3_example_3_2_func(x[tmp+i],y[tmp])])
			verts3.append([x[-1],y[tmp],0.])
		
	fig = plt.figure(figsize=(16, 8))
	if option == 'default':
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		#ax1.set_zlim([0,4])
		ax1.set_title('Surface')
		ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
		ax1.view_init(30, 325)
		ax2 = fig.add_subplot(122)
		ax2.set_xlabel('x')
		ax2.set_ylabel('y')
		ax2.set_title('Integration region')
		ax2.plot(intreg_x,intreg_y, 'k--')
		#ax2.plot([x[0],[-1]],[y[0],y[0]], 'k--o')
		#ax2.plot([x[-1],x[-1]],[y[0],y[-1]], 'k--o')
		#ax2.plot([x[-1],x[0]],[y[-1],y[0]], 'k--o')
		#r = Poly3DCollection([verts], linewidth=1, alpha=0.5)
		#r.set_facecolor((0.2, 0.2, 0.2, 0.5))
		#ax1.add_collection3d(r)
	else:
		ax1 = fig.add_subplot(111, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		#ax1.set_zlim([0,4])
		#ax1.plot_wireframe(X, Y, 0*Z, rstride=nxy, cstride=nxy, color='k', linewidth=1.0, antialiased=True)
		ax1.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=0.4)
		face1 = Poly3DCollection([verts1], linewidth=1, alpha=0.5)
		face2 = Poly3DCollection([verts2], linewidth=1, alpha=0.5)
		face3 = Poly3DCollection([verts3], linewidth=1, alpha=0.5)
		face1.set_facecolor((0, 0, 1, 0.5))
		face2.set_facecolor((0, 0, 1, 0.5))
		face3.set_facecolor((0, 0, 1, 0.5))
		ax1.add_collection3d(face1)
		ax1.add_collection3d(face2)
		ax1.add_collection3d(face3)
		ax1.view_init(30, 315)
		r = Poly3DCollection([verts], linewidth=1, alpha=0.5)
		r.set_facecolor((0.2, 0.2, 0.2, 0.5))
		ax1.add_collection3d(r)