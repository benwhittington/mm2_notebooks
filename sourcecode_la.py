# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# don't care about the complex numbers warning
import warnings

warnings.filterwarnings('ignore')


def example_1():
    n = 101
    x1 = np.linspace(0., 1.5, n)
    y1 = 2. * x1 - 1.
    x2 = np.linspace(0., 2., n)
    y2 = (3. - x2) / 2.

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    line1 = ax1.plot(x1, y1, 'r-', label='2x-y=1')
    line2 = ax1.plot(x2, y2, 'b-', label='x+2y=3')
    ax1.axhline(linewidth=2, color='k')
    ax1.axvline(linewidth=2, color='k')
    ax1.plot(1, 1, 'ko')
    ax1.plot([1, 1], [0, 1], 'k--')
    ax1.plot([0, 1], [1, 1], 'k--')
    ax1.legend(fontsize=20)


def jacobi_1(x0, y0, display):
    # just wanted to get rid of red underline
    xold, yold, xnew, ynew = 0, 0, 0, 0

    x = [x0]
    y = [y0]
    tol = 0.00005
    print(tol)
    tol = 5.e-5
    print(tol)
    print("Iteration xold    yold    xnew    ynew")
    for i in range(100):
        x[i + 1] = 0.5 * y[i] + 0.5
        y[i + 1] = -0.5 * x[i] + 1.5
        print("%02d        %6.4f  %6.4f  %6.4f  %6.4f" % (i + 1, xold, yold, xnew, ynew))
        if np.abs(x[i + 1] - 1.) <= tol and np.abs(y[i + 1] - 1.) <= tol:
            break

        ##########


# eigen

##########

def EigenTransform():
    """
		Controls UI. Called by notebook. Built on code from Colin Simpson
	"""

    # vec = [2./np.sqrt(5),1/np.sqrt(5)]
    # vec = [1./np.sqrt(2),1/np.sqrt(2)]

    transMatDict = {'[-0.307, 0.538], [2.666, -1.154]': 1, '[5, -2], [-2, 2]': 2, '[1.5, 0.25], [0.25, 1.5]': 3}

    origDict = {'Sq.1': 1, 'Sq.2:': 2, 'Outline': 3}

    mat_drp = widgets.Dropdown(options=transMatDict, description='Trsfrm Mat')
    orig_drp = widgets.Dropdown(options=origDict, description='Orig Shape')
    norm_tick = widgets.Checkbox(value=False, description='Normalise Eig')

    return widgets.VBox([
        widgets.HBox([
            mat_drp,
            orig_drp,
            norm_tick
        ]),
        widgets.interactive_output(showeigenstuff, {
            'ASelect': mat_drp,
            'origSelect': orig_drp,
            'normalised': norm_tick
        }
                                   )
    ])


def showeigenstuff(ASelect, origSelect, normalised):
    """ Built on Code from Colin Simpson.

		----------

		Parameters

		----------
		
		ASelect: int
			value to determine selection of transformation matrix

		origSelect: int
			value to determine selection of original shape

		normalised: bool
			true if displaying normalsed eig. False otherwise

	"""

    # select transMat. ipython doesn't want to let me pass in a function as part
    # of the dictionary that makes the dropdown so this is the best I can do atm

    if ASelect == 1:
        A = np.array([[-0.307, 0.538], [2.666, -1.154]])

    elif ASelect == 2:
        A = np.array([[5, -2], [-2, 2]])

    elif ASelect == 3:
        A = np.array([[1.5, 0.25], [0.25, 1.5]])

    # select original shape
    if origSelect == 1:
        a = 1. / np.sqrt(2)  # basic length of box
        orig_x = np.array([0 * a, 1. * a, 1. * a, 0 * a, 0 * a])
        orig_y = np.array([0 * a, 0 * a, 1. * a, 1. * a, 0 * a])

    elif origSelect == 2:
        a = 1.  # basic length of box
        orig_x = np.array([-1. * a, 1. * a, 1. * a, -1. * a, -1. * a])
        orig_y = np.array([-1. * a, -1. * a, 1. * a, 1. * a, -1. * a])

    # CCS select outline from external text file
    elif origSelect == 3:
        outline = np.loadtxt('rupiXY.txt')
        orig_x = outline[:, 0]
        orig_y = outline[:, 1]

    display(
        widgets.interactive_output(
            computeTransform,
            {
                'A': widgets.fixed(A),
                'orig_x': widgets.fixed(orig_x),
                'orig_y': widgets.fixed(orig_y),
                'normalised': widgets.fixed(normalised),
            }
        )
    )


def Afunc(k):
    k *= 2 * np.pi / 360

    A = np.array([
        [np.cos(k), -np.sin(k)],
        [np.sin(k), np.cos(k)]
    ])
    return A


def computeTransform(A, orig_x, orig_y, normalised):
    """
		Performs maths to compute transform and eigens. Code from Colin Simpson

		----------

		Parameters

		----------

		A: array_like
			transformation matrix

		orig_x: array_like
			original points of untransformed shape (x)

		orig_y: array_like
			original points of untransformed shape (y)
		
		normalised: bool
			True if eigs should be normalised
	"""
    w, v = np.linalg.eig(A)

    # extract eigens. Note that eigen1/2 contain eigen VECTORS
    eigen1 = v[:, 0]
    eigen2 = v[:, 1]
    value1 = w[0]
    value2 = w[1]

    deform_x = np.copy(orig_x)
    deform_y = np.copy(orig_y)

    for i in range(len(orig_x)):
        orig = np.array([orig_x[i], orig_y[i]])
        deform = np.matmul(A, orig)
        deform_x[i] = deform[0]
        deform_y[i] = deform[1]

    plotEigen(eigen1, eigen2, value1, value2, orig_x, orig_y, deform_x, deform_y, normalised)


#     if show_vec:
#         ax1.plot([0.,vline[0]],[0.,vline[1]] , 'g--o', label='Deformed vector')
#         ax1.plot([0.,vec[0]],[0.,vec[1]] , 'k-o', label='Original vector')

def plotEigen(eigen1, eigen2, value1, value2, orig_x, orig_y, deform_x, deform_y, normalised):
    """
		Responsible for plotting of all the eigen stuff

		----------

		Parameters

		----------

		eigen1: array_like
			first eigen vector

		eigen2: array_like
			second eigen vector

		value1: float
			first eigen value

		value2: flaot
			second eigen value
		
		orig_x: array_like
			x ordinates of original shape

		orig_y: array_like
			y ordinates of original shape

		deform_x: array_like
			x ordinates of deformed shape

		deform_y: array_like
			y ordinates of deformed shape

		normalised: bool
			true if displaying normalsed eig. False otherwise
		
	"""

    lim = np.max([np.max(np.abs(eigen1)), np.max(np.abs(eigen2)), np.max(np.abs(orig_x)), np.max(np.abs(orig_y)),
                  np.max(np.abs(deform_x)), np.max(np.abs(deform_y))])

    # lim = np.maximum(np.abs(eigen1), np.abs(eigen2), np.abs(orig_x), np.abs(orig_y), np.abs(deform_x), np.abs(deform_y))

    # need to remove all the non-plotting parts of this func
    _, ax1 = plt.subplots(1, 1, figsize=(12, 12))

    print("Eigen Vector 1: {}, Eigen Value 1: {}".format(eigen1, value1))
    print("Eigen Vector 2: {}, Eigen Value 2: {}".format(eigen2, value2))

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.axhline(linewidth=1, color='k')
    ax1.axvline(linewidth=1, color='k')

    ax1.plot(orig_x, orig_y, 'k--o', label='Original Shape')
    ax1.plot(deform_x, deform_y, 'g--o', label='Deformed Shape')

    eigen1_x = np.array([-eigen1[0], eigen1[0]])
    eigen1_y = np.array([-eigen1[1], eigen1[1]])

    eigen2_x = np.array([-eigen2[0], eigen2[0]])
    eigen2_y = np.array([-eigen2[1], eigen2[1]])

    if normalised:
        ax1.plot(eigen1_x, eigen1_y, 'r-o', label=r'$\vec{s}_1$')
        ax1.plot(eigen2_x, eigen2_y, 'b-o', label=r'$\vec{s}_2$')

    elif not normalised:
        ax1.plot(eigen1_x * value1, eigen1_y * value1, 'r--o', label=r'$\lambda_1 \vec{s}_1$')
        ax1.plot(eigen2_x * value2, eigen2_y * value2, 'b--o', label=r'$\lambda_2 \vec{s}_2$')

    ax1.legend(fontsize=12)

    ax1.set_xlim(-lim + 0.1 * lim, lim + 0.1 * lim)
    ax1.set_ylim(-lim + 0.1 * lim, lim + 0.1 * lim)


##########

#  matrices

###########

def CoordinateTransform(custA=np.array([])):
    """
		Main function for UI. Called by notebook

		----------

		Parameters

		----------

		custA: numpy.ndarray
			Optinal matrix passed in by user
	"""

    transOptions = ['Scale', 'Stretch X', 'Stretch Y', 'Shear X', 'Shear Y', 'Rotation', 'Custom', 'All']
    shapeOptions = {'Square': 1, 'Triangle': 2, 'Arrow': 3}

    shape_drop = widgets.Dropdown(options=shapeOptions)
    trans_drop = widgets.Dropdown(options=transOptions)
    showMat_check = widgets.Checkbox(description='Show Mat.')
    showEig_check = widgets.Checkbox(description='Show Eigs')
    normEig_check = widgets.Checkbox(description='Normalise Eigs')
    matWid_html = widgets.HTML(value='')

    display(widgets.VBox([
        widgets.HBox([
            shape_drop,
            trans_drop,
            showMat_check,
            showEig_check,
            normEig_check
        ]),
        matWid_html,
        widgets.interactive_output(matrixVarSlider, {
            'shapeType': shape_drop,
            'transType': trans_drop,
            'custA': widgets.fixed(custA),
            'showMat': showMat_check,
            'matWid_html': widgets.fixed(matWid_html),
            'showEigs': showEig_check,
            'normEig': normEig_check
        })
    ]))


def returnSlider(transType):
    """ Returns slider based on transformation type and parameters required

		----------

		Parameters

		----------

		transType: string
			for identifying user selected transformation

		----------

		Returns

		----------

		ipython slider object
	"""

    #  pretty ugly way of doing this. Wanted to only create each slider in one place
    # and have the option to return all of them
    scale_slider = widgets.FloatSlider(value=1, min=0, max=2, step=0.1, description='Scale', continuous_update=False)
    stretchX_slider = widgets.IntSlider(value=1, min=-5, max=5, description='Stretch X', continuous_update=False)
    shearX_slider = widgets.IntSlider(value=0, min=-5, max=5, description='Shear X', continuous_update=False)
    stretchY_slider = widgets.IntSlider(value=1, min=-5, max=5, description='Stretch Y', continuous_update=False)
    shearY_slider = widgets.IntSlider(value=0, min=-5, max=5, description='Shear Y', continuous_update=False)
    rot_slider = widgets.IntSlider(value=0, min=-180, max=180, step=5, description='Angle (deg)',
                                   continuous_update=False)

    if transType == 'Scale':
        slider = scale_slider
    elif (transType == 'Stretch X'):
        slider = stretchX_slider
    elif (transType == 'Stretch Y'):
        slider = stretchY_slider
    elif (transType == 'Shear X'):
        slider = shearX_slider
    elif (transType == 'Shear Y'):
        slider = shearY_slider
    elif transType == 'Rotation':
        slider = rot_slider

    elif transType == 'Custom':
        slider = widgets.fixed([])

    elif transType == "All":
        slider = [
            scale_slider,
            stretchX_slider,
            stretchY_slider,
            shearX_slider,
            shearY_slider,
            rot_slider
        ]

    return slider


def matrixVarSlider(shapeType, transType, custA, showMat, showEigs, normEig, matWid_html):
    """
		Intermediate function needed to produce k slider

		----------

		Parameters

		----------

		shapeType: string
			for identifying user selected shape

		transType: string
			for identifying user selected transformation

		custA: numpy.ndarrray
			Optional transformation matrix passed in by user

		showMat: bool
			True if printing transformation matrix

		showEigs: bool
			True of printing eigs

		normEig: bool
			True if displayed eigs should be normalised

		matWid_html: ipywidgets object
			html widget for displaying matrix in

		----------

		Returns

		----------

		Ipython display object
	"""

    var_sldr = returnSlider(transType)

    # choose shape
    # Current version of ipython (7.1.1) won't allow arrays passed into shapeType from dict dropdown
    if shapeType == 1:
        pointsXY = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    elif shapeType == 2:
        pointsXY = np.array([[-1, 0, 1, -1], [0, 1, 0, 0]])
    elif shapeType == 3:
        pointsXY = np.array([[-1, -1, -2, 0, 2, 1, 1, -1], [0, 2, 2, 4, 2, 2, 0, 0]])

    if transType == 'Custom':
        return display(
            widgets.interactive_output(matrixMath, {
                'pointsXY': widgets.fixed(pointsXY),
                'transType': widgets.fixed(transType),
                'k': var_sldr,
                'custA': widgets.fixed(custA),
                'showMat': widgets.fixed(showMat),
                'matWid_html': widgets.fixed(matWid_html),
                'showEigs': widgets.fixed(showEigs),
                'normEig': widgets.fixed(normEig)
            })
        )

    elif transType == 'All':
        return display(
            widgets.HBox([
                widgets.VBox([
                    var_sldr[0],
                    var_sldr[1]
                ]),
                widgets.VBox([
                    var_sldr[2],
                    var_sldr[3],
                ]),
                widgets.VBox([
                    var_sldr[4],
                    var_sldr[5]
                ])
            ]),
            widgets.interactive_output(matrixMath, {
                'pointsXY': widgets.fixed(pointsXY),
                'transType': widgets.fixed(transType),
                'custA': widgets.fixed(custA),
                'showMat': widgets.fixed(showMat),
                'matWid_html': widgets.fixed(matWid_html),
                'showEigs': widgets.fixed(showEigs),
                'normEig': widgets.fixed(normEig),
                'Scale': var_sldr[0],
                'Stretch X': var_sldr[1],
                'Stretch Y': var_sldr[2],
                'Shear X': var_sldr[3],
                'Shear Y': var_sldr[4],
                'Rotation': var_sldr[5]
            })
        )

    # normal k, no custom, no all
    return display(
        var_sldr,
        widgets.interactive_output(matrixMath, {
            'pointsXY': widgets.fixed(pointsXY),
            'transType': widgets.fixed(transType),
            'k': var_sldr,
            'custA': widgets.fixed(custA),
            'showMat': widgets.fixed(showMat),
            'matWid_html': widgets.fixed(matWid_html),
            'showEigs': widgets.fixed(showEigs),
            'normEig': widgets.fixed(normEig)
        })
    )


def printMat(mat, matWid_html):
    """
		Formats mat as HTML table to be outputted using ipython.widgets.HTML

		----------

		Parameters

		----------

		mat: array_like
			matrix to be formatted

		matWid_html: ipywidgets object
			html widget for displaying matrix in

	"""
    if type(mat) == np.ndarray:
        matWid_html.value = '<font size="+2"><table style="width:20%"><tr><td>{:.3f}</td><td>{:.3f}</td></tr><tr><td>{:.3f}</td><td>{:.3f}</td></tr></font>'.format(
            float(mat[0][0]), float(mat[0][1]), float(mat[1][0]), float(mat[1][1]))

    else:
        matWid_html.value = ''


def matrixMath(pointsXY, transType, custA, showMat, showEigs, normEig, matWid_html, **kwargs):
    """
		From Colin Simpson's code. Computes eigs and 

		----------

		Parameters

		----------

		pointsXY: array_like
			untransformed shape points

		shapeType: string
			for identifying user selected shape

		transType: string
			for identifying user selected transformation

		custA: numpy.ndarrray
			Optional transformation matrix passed in by user

		k: float
			variable for transformation parameter magnitude
		
		showMat: bool
			True if printing transformation matrix

		showEigs: bool
			True of printing eigs

		normEig: bool
			True if displayed eigs should be normalised

		matWid_html: ipywidgets object
			html widget for displaying matrix in
	"""

    if len(kwargs) == 1:
        k = kwargs['k']
        transform = chooseTransform(transType, k, custA)

    elif len(kwargs) == 6:

        transform = np.array([[1, 0], [0, 1]])

        for key, k in kwargs.items():
            out = chooseTransform(key, k, custA)
            transform = np.matmul(transform, out)

    if showMat:
        printMat(transform, matWid_html)

    else:
        printMat('', matWid_html)

    # assert valid matrix has been passed in when using custom option
    assert (not (transform.shape != (
    2, 2) and transType == 'Custom')), 'Pass in a valid transformation matrix to use custom option'

    # define and perform transformation
    pointsUV = np.matmul(transform, pointsXY)

    # vectors represeting x, y, u, v unit vectors
    vectorX = np.array([1, 0])
    vectorY = np.array([0, 1])
    vectorU = np.matmul(transform, vectorX)
    vectorV = np.matmul(transform, vectorY)

    # pack vectors
    vectors = [vectorU, vectorV]

    if (transType == 'Stretch Y') and (len(pointsXY[0]) == 8) and (k == -1):
        colour = 'C0'
    else:
        colour = 'r'

    squareUV = plt.Polygon(pointsUV.transpose(), color=colour, alpha=0.5, edgecolor='k')
    squareXY = plt.Polygon(pointsXY.transpose(), color='C0', alpha=0.1, edgecolor='k')
    # squareXY.set_alpha(0.25)

    w, v = np.linalg.eig(transform)

    eigen1 = v[:, 0]
    eigen2 = v[:, 1]

    value1 = w[0]
    value2 = w[1]

    plotMatrices(squareXY, squareUV, vectors, eigen1, eigen2, value1, value2, showEigs, normEig)


def plotMatrices(squareXY, squareUV, vectors, eigen1, eigen2, value1, value2, showEigs, normEig, ):
    """
		Plotting function for matrices

		----------

		Parameters

		----------

		squareXY: matplotlib object
			untransformed xy square

		squareUV: matplotlib object
			transformed xy->uv square

		vectors: array_like
			array of x,y,u,v unit vectors)

	"""

    buf = 0.1
    asc = 0.1  # arrow scale
    origin = np.array([0, 0])

    vectorU, vectorV = vectors

    _, ax = plt.subplots(1, 1, figsize=(12, 12))

    eigen1_x = np.array([-eigen1[0], eigen1[0]])
    eigen1_y = np.array([-eigen1[1], eigen1[1]])

    eigen2_x = np.array([-eigen2[0], eigen2[0]])
    eigen2_y = np.array([-eigen2[1], eigen2[1]])

    if normEig and showEigs:
        ax.plot(eigen1_x, eigen1_y, 'r-o', label=r'$\vec{s}_1$')
        ax.plot(eigen2_x, eigen2_y, 'b-o', label=r'$\vec{s}_2$')

    elif (not normEig) and (showEigs):
        ax.plot(eigen1_x * value1, eigen1_y * value1, 'r-o', label=r'$\lambda_1 \vec{s}_1$')
        ax.plot(eigen2_x * value2, eigen2_y * value2, 'b-o', label=r'$\lambda_2 \vec{s}_2$')

    ax.add_patch(squareUV)
    ax.add_patch(squareXY)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim([-5. - buf, 5. + buf])
    ax.set_ylim([-5. - buf, 5. + buf])
    ax.axhline(linewidth=0.5, color='k')
    ax.axvline(linewidth=0.5, color='k')
    ax.arrow(*origin, *vectorU, fc='k', ec='k', length_includes_head=True, head_length=asc, head_width=asc)
    ax.arrow(*origin, *vectorV, fc='k', ec='k', length_includes_head=True, head_length=asc, head_width=asc)


def chooseTransform(transType, k, custA):
    """
		Returns correct transformation matrix based on transType

		----------

		Parameters

		----------

		Transtype: string
			key for selecting transformation matrix

		k: float
			transformation matrix value magnitude

		custA: array_like
			custom matrix from user

		----------

		Returns

		----------

		transform: array_like
			transformation matrix
	"""

    # fudge factor to stop things breaking when k is 0
    if k == 0: k = 0.0001

    if transType == 'Scale':
        transform = np.array([
            [k, 0],
            [0, k]
        ])

    elif transType == 'Stretch X':
        transform = np.array([
            [k, 0],
            [0, 1]
        ])

    elif transType == 'Stretch Y':
        transform = np.array([
            [1, 0],
            [0, k]
        ])

    elif transType == 'Shear X':
        transform = np.array([
            [1, k],
            [0, 1]
        ])

    elif transType == 'Shear Y':
        transform = np.array([
            [1, 0],
            [k, 1]
        ])

    elif transType == 'Rotation':
        k *= 2 * np.pi / 360
        transform = np.array([
            [np.cos(k), -np.sin(k)],
            [np.sin(k), np.cos(k)]
        ])

    elif transType == 'Custom':
        transform = custA

    elif transType == 'All':
        pass

    return transform


##########

# elipses

##########

def EllipseTransform():
    """
		UI. Called by notebook
	"""
    eqns = {'[1./4., 0.], [0., 1./9.]': 'Q1', '[5, -2], [-2, 2]': 'Q2', '[1./4., 0], [0, 1./4.]': 'Q3'}

    eqn_drp = widgets.Dropdown(options=eqns)
    printEigs_check = widgets.Checkbox(value=False, description='Print Eigs')

    return widgets.VBox([
        widgets.HBox([
            eqn_drp,
            printEigs_check
        ]),
        widgets.interactive_output(ellipseMath, {
            'qSelect': eqn_drp,
            'k': widgets.fixed(1),
            'printEigs': printEigs_check
        })
    ])


def ellipseMath(qSelect, k, printEigs):
    """
		Main function for producing ellipses. Code from Colin Simpson
		
		----------

		Parameters

		----------

		qSelect: string
			string for selecting equaton (if you're looking at this after Jan 2019 see if ipython lets allows functions in dropdown dict because it's bugging me)

		k: float
			Can't remember what this is. Ask Colin

	"""

    if qSelect == 'Q1':
        Q = np.array([[1. / 4., 0.], [0., 1. / 9.]])
    elif qSelect == 'Q2':
        Q = np.array([[5, -2], [-2, 2]])
    elif qSelect == 'Q3':
        Q = np.array([[1. / 4., 0], [0, 1. / 4.]])

    # find eigenvalues/eigenvectors and eigenvector matrix
    w, v = np.linalg.eig(Q)
    S = np.array([v[:, 1], v[:, 0]])

    if printEigs:
        v1 = v[:, 1]
        v2 = v[:, 0]
        print('Eigen Vector 1: {}, Eigen Value 1: {}'.format(v1, w[1]))
        print('Eigen Vector 2: {}, Eigen Value 2: {}'.format(v2, w[0]))

    # find s1s2 coordinates of vertices and covertices
    a = np.sqrt(k / w[1])  # semimajor axis length
    b = np.sqrt(k / w[0])  # semiminor axis length

    # coordinates of vertices in eigenvector domain
    vertsEig = np.array([
        [-1. * a, 0],
        [1. * a, 0],
        [0, -1. * b],
        [0, 1. * b]
    ])

    # find xy coordinates of vertices and covertices
    vertsXY = np.array([
        np.matmul(S, vertsEig[0]),
        np.matmul(S, vertsEig[1]),
        np.matmul(S, vertsEig[2]),
        np.matmul(S, vertsEig[3])
    ])

    # find rotation angle of ellipse from one of the vertices
    t_rot = np.arctan2(vertsXY[1][1], vertsXY[1][0])

    # find ellipse in s1s2 coordinate system
    t = np.linspace(0, 2 * np.pi, 1000)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])

    # define rotation matrix
    R_rot = np.array([[np.cos(t_rot), -np.sin(t_rot)], [np.sin(t_rot), np.cos(t_rot)]])  # 2-D rotation matrix

    # apply rotation to find ellipse in xy coordinate system
    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])

    plotElipses(vertsEig, vertsXY, Ell, Ell_rot, v)


def plotElipses(vertsEig, vertsXY, Ell, Ell_rot, v):
    """
		Main plotting function for ellipse. Code from Colin Simpson

		----------

		Parameters

		----------

		vertsEig: array_like
			Vertices in eigen coordinate system
		
		vertsXY: array_like
			verts in XY corrds

		Ell: array_like
			points for ellipse

		Ell_rot: array_like
			points for ellipse
	"""

    #  needs tidying up

    b1, b2, b3, b4 = vertsEig
    i1, i2, i3, i4 = vertsXY
    origin = [0, 0]
    asc = 0.1

    # plot the initial ellipse on the left
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)

    ax1.plot(Ell[0, :], Ell[1, :], 'k-')

    ax1.plot(b1[0], b1[1], 'black', marker='.', markersize=10, linestyle='None')
    ax1.plot(b2[0], b2[1], 'black', marker='.', markersize=10, linestyle='None')
    ax1.plot(b3[0], b3[1], 'black', marker='.', markersize=10, linestyle='None')
    ax1.plot(b4[0], b4[1], 'black', marker='.', markersize=10, linestyle='None')

    ax1.set_xlabel(r'$s_1$', fontsize=14)
    ax1.set_ylabel(r'$s_2$', fontsize=14)
    ax1.axhline(linewidth=0.5, color='k')
    ax1.axvline(linewidth=0.5, color='k')

    ax1.plot([0, b1[0]], [0, b1[1]], 'k--')
    ax1.plot([0, b3[0]], [0, b3[1]], 'k--')

    ax1.arrow(*origin, *b2, fc='r', ec='r', length_includes_head=True, head_length=asc, head_width=asc,
              label=r'$\lambda_2 \vec{s}_2$')
    ax1.arrow(*origin, *b4, fc='r', ec='r', length_includes_head=True, head_length=asc, head_width=asc,
              label=r'$\lambda_1 \vec{s}_1$')

    # plot the rotated ellipse on the right
    ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)

    ax2.plot(Ell_rot[0, :], Ell_rot[1, :], 'k-')
    ax2.plot([0, i1[0]], [0, i1[1]], 'k--')
    ax2.plot([0, i3[0]], [0, i3[1]], 'k--')

    ax2.arrow(*origin, *i2, fc='r', ec='r', length_includes_head=True, head_length=asc, head_width=asc,
              label=r'$\lambda_2 \vec{s}_2$')
    ax2.arrow(*origin, *i4, fc='r', ec='r', length_includes_head=True, head_length=asc, head_width=asc,
              label=r'$\lambda_1 \vec{s}_1$')

    ax2.plot(i1[0], i1[1], 'black', marker='.', markersize=10, linestyle='None')
    ax2.plot(i2[0], i2[1], 'black', marker='.', markersize=10, linestyle='None')
    ax2.plot(i3[0], i3[1], 'black', marker='.', markersize=10, linestyle='None')
    ax2.plot(i4[0], i4[1], 'black', marker='.', markersize=10, linestyle='None')

    ax2.set_xlabel(r'$x$', fontsize=14)
    ax2.set_ylabel(r'$y$', fontsize=14)
    ax2.axhline(linewidth=0.5, color='k')
    ax2.axvline(linewidth=0.5, color='k')


##########

# iterative

##########

# following equations are for iterative methods to solve 

def f1_2D(x, y):
    return y / 2 + 0.5


def f2_2D(x, y):
    return -x / 2 + 3 / 2


def f3_2D(x, y):
    return -3 * x + 2


def f4_2D(x, y):
    return y + 3


def f1_3D(x1, x2, x3):
    """
		x1 dependant
		from example 2.1 in iterative methods
	"""
    return -0.2 * x2 - 0.2 * x3 + 0.8


def f2_3D(x1, x2, x3):
    """
		x2 dependant
		from example 2.1 in iterative methods
	"""
    return -0.1 * x1 - 0.2 * x3 - 0.1


def f3_3D(x1, x2, x3):
    """
		x3 dependant
		from example 2.1 in iterative methods
	"""
    return -0.1 * x1 - 0.1 * x2 - 0.9


def f4_3D(x1, x2, x3):
    return 0.4 * x2 + 0.6 * x3 + 0.8


def f5_3D(x1, x2, x3):
    return -0.2 * x1 + 0.8 * x3 + 1.5


def f6_3D(x1, x2, x3):
    return -0.2 * x1 + 0.2 * x2 + 1.23


def jacobi(init, eqns, tol=1, step=20):
    """
		Runs Jacobi's method of solving systems of linear equations

		----------

		Parameters

		----------

		n: int
			number of linear equations (rows in matrix). should be of len n

		init: array_like
			initial conditions for each equation

		eqns: array_like
			list of function handles corresponding to each equation. Should be of len n.
			Each funcion must take n dependant variables as input, even if not all of them are used

		tol: float, optional
			Maximum allowed change between iterations before termination

		----------

		Returns

		----------

		out: array_like
			n by k array where n is the number of equations/variables and n is the required iterations

		count: int
			number of iterations required

	"""
    n = len(init)
    assert (len(eqns) == n), "Must be same number of initial conditions as equations"

    out = np.array([init])
    nextIt = np.zeros(n)
    sumSquares = 1
    count = 0
    while count < step:
        for j in range(n):
            # loop each eqn
            # unpack last iteration values into each equation.
            # put returns from eqns into nextIt
            nextIt[j] = eqns[j](*(out[-1]))

        out = np.append(out, [nextIt], 0)

        sumSquares = np.sum((out[-1] - out[-2]) ** 2)
        count += 1

    return out, count


def gaussSeidel(init, eqns, tol=1, step=20):
    """
		Performs Gauss Seidel method of solving systems of linear equations

		----------

		Parameters

		----------

		n: int
			number of linear equations (rows in matrix). should be of len n

		init: array_like
			initial conditions for each equation

		eqns: array_like
			list of function handles corresponding to each equation. Should be of len n.
			Each funcion must take n dependant variables as input, even if not all of them are used

		tol: float, optional
			Maximum allowed change between iterations before termination

		----------

		Returns

		----------

		out: array_like
			n by k array where n is the number of equations/variables and n is the required iterations

		count: int
			number of iterations required
	"""

    n = len(init)
    assert (len(eqns) == n), "Must be same number of initial conditions as equations"

    out = np.array([init])
    nextIt = np.zeros(n)
    count = 0
    sumSquares = 1
    while count < step:
        nextIt[0] = eqns[0](*(out[-1]))
        # print(n-1)
        for j in range(1, n):
            nextIt[j] = eqns[j](*np.append(nextIt[:j], out[-1][j:]))

        out = np.append(out, [nextIt], 0)

        count += 1
        sumSquares = np.sum((out[-1] - out[-2]) ** 2)

    return out, count


def plotIterative3D(ax, xks, label=None):
    """ Plots 3D gauss seidel jacobi comparison.
		(Not sure this is necessary to have dedicated function)

		----------

		Parameters

		----------

		ax: matplotlib object
			axis to plot on

		xks: array_like
			independant variables in each dimension

		label: string
			label for plot legend
	"""
    ax.plot(xs=xks[:, 0], ys=xks[:, 1], zs=xks[:, 2], label=label)


def iterative3D(tol, start, sysSelect=2, showIts=False, showSurfs=True):
    """ Main function for 3d interative methods. Does printing and calls plotting functions

		----------

		Parameters

		----------

		tol: float
			Maximum allowed change between iterations before termination

		start: array_like
			starting point for iteration. Must be of length 3

		sysSelect: int
			key for selecting system of equations

		showIts: bool
			For printing each iteration

		showSurfs: bool
			For displaying surfaces representing system of equations

	"""
    # analytic sln
    buf = 1

    if sysSelect == 1:
        eqn1 = f1_3D
        eqn2 = f2_3D
        eqn3 = f3_3D
    elif sysSelect == 2:
        eqn1 = f4_3D
        eqn2 = f5_3D
        eqn3 = f6_3D

    init = np.array(start)
    eqns = np.array([eqn1, eqn2, eqn3])

    x = np.arange(-1, 1, 0.1)

    jOut, jCount = jacobi(init, eqns, tol=tol)
    gOut, gCount = gaussSeidel(init, eqns, tol=tol)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    xk, yk, zk = gOut[-1]

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')

    # use solution found to set axes
    ax.set_xlim(xk - buf, xk + buf)
    ax.set_ylim(yk - buf, yk + buf)
    ax.set_zlim(zk - buf, zk + buf)

    plotIterative3D(ax, jOut, label='Jacobi')
    plotIterative3D(ax, gOut, label='Gauss Seidel')

    if showSurfs:
        # need pairs for mesh grid so domain is correct. Only need 2 points for meshgrid
        XY, YX = np.meshgrid(np.linspace(xk - buf, xk + buf, 2), np.linspace(yk - buf, yk + buf, 2))
        YZ, ZY = np.meshgrid(np.linspace(yk - buf, yk + buf, 2), np.linspace(zk - buf, zk + buf, 2))
        XZ, ZX = np.meshgrid(np.linspace(yk - buf, yk + buf, 2), np.linspace(zk - buf, zk + buf, 2))

        x1 = eqn1(None, YZ, ZY)
        x2 = eqn2(XZ, None, ZX)
        x3 = eqn3(XY, YX, None)

        alpha = 0.2  # opacity

        # cant label surfaces without an error for some reason
        ax.plot_surface(x1, YZ, ZY, alpha=alpha, color='r')
        ax.plot_surface(XZ, x2, ZX, alpha=alpha, color='g')
        ax.plot_surface(XY, YX, x3, alpha=alpha, color='b')

    print("Jacobi iterations required:       ", jCount)
    print("Gauss Seidel iterations required: ", gCount)
    print("Jacobi final iteration:       x1: {:10.7e}, x2: {:10.7e}, x3: {:10.7e}".format(*jOut[-1]))
    print("Gauss Seidel final iteration: x1: {:10.7e}, x2: {:10.7e}, x3: {:10.7e}".format(*gOut[-1]))

    if showIts:
        print("\n\nJacobi: ")
        print(jOut)
        print("\n\nGauss Seidel")
        print(gOut)

    ax.legend()
    plt.show()


def IterativeMethods3D(start=[0, 0, 0]):
    """
		Method called by notebook. Produces UI and calls iterative functions

		Parameters

		----------

		start: array_like
			starting points for iteration

	"""
    eqnOpt = {'Sys. 1': 1, 'Sys. 2': 2}

    tol_slider = widgets.FloatLogSlider(0.1, min=-10, max=-1, step=1, description='Tolerance', continuous_update=False)
    printOut_check = widgets.Checkbox(value=False, description='Show Iterations')
    showSurf_check = widgets.Checkbox(value=False, description='Show Surfaces')
    eqns_drop = widgets.Dropdown(options=eqnOpt)

    display(widgets.VBox([
        widgets.HBox([
            tol_slider,
            eqns_drop,
            printOut_check,
            showSurf_check
        ]),
        widgets.interactive_output(iterative3D, {
            'tol': tol_slider,
            'showIts': printOut_check,
            'sysSelect': eqns_drop,
            'showSurfs': showSurf_check,
            'start': widgets.fixed(start)
        })
    ]))


def IterativeMethods2D(start=[0, 0]):
    """
		Main function called by notebook and producing UI for 2D iteration

		Parameters

		----------

		start: array_like
			starting points for iteration

	"""
    eqnOpt = {'Sys. 1': 1, 'Sys. 2': 2}
    step_slider = widgets.IntSlider(0, min=0, max=20, step=1, description='Step', continuous_update=False)
    printOut_check = widgets.Checkbox(value=False, description='Show Iterations')
    eqns_drop = widgets.Dropdown(options=eqnOpt)

    display(widgets.VBox([
        widgets.HBox([
            step_slider,
            eqns_drop,
            printOut_check
        ]),
        widgets.interactive_output(iterative2D, {
            'step': step_slider,
            'start': widgets.fixed(start),
            'sysSelect': eqns_drop,
            'showIts': printOut_check
        })
    ])
    )


def plotIterative2D(ax, xks, style='-', label=''):
    """
		Main function to handle 2D plotting

		Parameters

		----------

		ax: matplotlib object
			axis to plot on

		xks: array_like
			values from iteration to plot

		style: string
			matplotlib plotting style key

		label: string
			for label of plotted values

	"""
    ax.plot(xks[:, 0], xks[:, 1], style, label=label)


def iterative2D(step, start, sysSelect, showIts):
    """
		Main function for 2D iterative methods. Calls iterative solvers and does printing/plotting

		----------

		Parameters

		----------

		tol: float
			maximum allowed difference between iterations before termination

		start: array_like
			starting values

		sysSelect: int
			key for selecting system of equations

		showIts: bool
			for printing each iteration

	"""

    # use dictionary?
    if sysSelect == 1:
        eqn1 = f1_2D
        eqn2 = f2_2D

    elif sysSelect == 2:
        eqn1 = f3_2D
        eqn2 = f4_2D

    buf = 2

    jOut, jCount = jacobi(start, [eqn1, eqn2], step=step)
    gOut, gCount = gaussSeidel(start, [eqn1, eqn2], step=step)

    if showIts:
        print("\n\nJacobi: ")
        print(jOut)
        print("\n\nGauss Seidel")
        print(gOut)

    _, ax = plt.subplots(1, 1, figsize=(16, 8))

    plotIterative2D(ax, jOut, style='--o', label='Jacobi')
    plotIterative2D(ax, gOut, style='--o', label='Gauss Seidel')

    # x0, x1 = gOut[-1]

    # x = np.linspace(x0-buf, x0+buf, 2)
    # y = np.linspace(x1-buf, x1+buf, 2)

    x = np.linspace(-4, 4, 2)
    y = np.linspace(-4, 4, 2)

    ax.plot(x, eqn1(x, y))
    ax.plot(x, eqn2(x, y))

    # ax.set_xlim(x0-buf, x0+buf)
    # ax.set_ylim(x1-buf, x1+buf)

    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)

    ax.legend()


if __name__ == "__main__":
    start = [-2, -2, -2]
    iterative3D(start=start, tol=1.e-4)
