import numpy as np

def gradapprC(g):
    n = 10**-6
    return lambda x: np.array([[(g(x + n * np.eye(len(x))[j])[i] - g(x - n * np.eye(len(x))[j])[i]) / (2*n)
                                  for j in range(len(x))]
                                 for i in range(len(g(x)))])

# ─── FUNZIONI PROVA ─────────────────────────────────────────────────────────
def initpointprova(n):
    x = np.zeros(n)
    x[0] = 20; x[1] = 20
    return x

def fprova():
    """Rastrigin 2D — minimo globale in (0,0) con f=0."""
    return lambda x: 20.0 + x[0]**2 - 10.0*np.cos(2*np.pi*x[0]) \
                          + x[1]**2 - 10.0*np.cos(2*np.pi*x[1])

def grad_fprova():
    """Gradiente della Rastrigin 2D."""
    return lambda x: np.array([
        2.0*x[0] + 20.0*np.pi*np.sin(2.0*np.pi*x[0]),
        2.0*x[1] + 20.0*np.pi*np.sin(2.0*np.pi*x[1])
    ])

# ─── FUNZIONI PER PENALITÀ SEQUENZIALI ──────────────────────────────────────
def initpoint1(n):
    x = np.zeros(n); x[0]=4.9; x[1]=0.1
    return np.array(x)
def f1():
    return lambda x: (x[0]-5)**2 + x[1]**2 - 25
def g1():
    return lambda x: [x[0]**2 - x[1]]
def h1():
    return lambda x: []
def gradientf1():
    return lambda x: np.array([2*(x[0]-5), 2*x[1]])
def gradientg1():
    return lambda x: [[2*x[0], -1]]
def gradienth1():
    return lambda x: []

def initpoint2(n):
    x = np.zeros(n); x[0]=20.1; x[1]=5.84
    return np.array(x)
def f2():
    return lambda x: (x[0]-10)**3 + (x[1]-20)**3
def g2():
    return lambda x: [-(x[0]-5)**2-(x[1]-5)**2+100,
                       (x[1]-5)**2+(x[0]-6)**2-82.81,
                       x[0]-100, 13-x[0], x[1]-100, -x[1]]
def h2():
    return lambda x: []
def gradientf2():
    return lambda x: np.array([3*(x[0]-10)**2, 3*(x[1]-20)**2])
def gradientg2():
    return lambda x: [[-2*(x[0]-5), -2*(x[1]-5)],
                       [2*(x[0]-6), 2*(x[1]-5)],
                       [1,0], [-1,0], [0,1], [0,-1]]
def gradienth2():
    return lambda x: []

def initpoint3(n):
    x = np.ones(n)
    return np.array(x)
def f3():
    return lambda x: x[0]**2 + x[1]**2 + x[2]**2
def g3():
    return lambda x: [-(x[0])**2-(x[1])**2+1,
                       x[0]-10, 1-x[0], x[1]-10, -x[1]-10, x[2]-10, -x[2]-10]
def h3():
    return lambda x: []
def gradientf3():
    return lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])
def gradientg3():
    return lambda x: [[-2*x[0],-2*x[1],0],[1,0,0],[-1,0,0],
                       [0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
def gradienth3():
    return lambda x: []

def initpoint4(n):
    x = np.zeros(n); x[0]=2; x[1]=2; x[2]=2
    return np.array(x)
def f4():
    return lambda x: (x[0]-1)**2 + (x[0]-x[1])**2 + (x[1]-x[2])**4
def g4():
    return lambda x: [x[0]-10,-x[0]-10,x[1]-10,-x[1]-10,x[2]-10,-x[2]-10]
def h4():
    return lambda x: [x[0]*(1+x[1]**2) + x[2]**4 - 4 - 3*np.sqrt(2)]
def gradientf4():
    return lambda x: [2*(x[0]-1)+2*x[0]-2*x[1],
                       2*x[1]-2*x[0]+4*(x[1]-x[2])**3,
                       -4*(x[1]-x[2])**3]
def gradientg4():
    return lambda x: [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
def gradienth4():
    return lambda x: [[1+x[1]**2, 2*x[0]*x[1], 4*x[2]**3]]

def initpoint5(n):
    x = np.zeros(n); x[0]=-5; x[1]=5; x[2]=0
    return np.array(x)
def f5():
    return lambda x: (x[0]-x[1])**2 + (x[0]+x[1]-10)**2/9 + (x[2]-5)**2
def g5():
    return lambda x: [-48+x[0]**2+x[1]**2+x[2]**2,
                       x[0]-4.5,-x[0]-4.5,x[1]-4.5,-x[1]-4.5,x[2]-5,-x[2]-5]
def h5():
    return lambda x: []
def gradientf5():
    return lambda x: [2*x[0]-2*x[1]+2/9*x[0]+2/9*x[1]-20/9,
                       2*x[1]-2*x[0]+2/9*x[1]+2/9*x[0]-20/9,
                       2*(x[2]-5)]
def gradientg5():
    return lambda x: [[2*x[0],2*x[1],2*x[2]],[1,0,0],[-1,0,0],
                       [0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
def gradienth5():
    return lambda x: []

def initpoint6(n):
    x = np.full(n, 0.5)
    return np.array(x)
def f6():
    return lambda x: (x[0]**2 + 0.5*x[1]**2 + x[2]**2 + 0.5*x[3]**2
                      - x[0]*x[2] + x[2]*x[3] - x[0] - 3*x[1] + x[2] - x[3])
def g6():
    return lambda x: [-5+x[0]+2*x[1]+x[2]+x[3],
                       -4+3*x[0]+x[1]+2*x[2]-x[3],
                       -x[1]-4*x[2]+1.5,
                       -x[0],-x[1],-x[2],-x[3]]
def h6():
    return lambda x: []
def gradientf6():
    return lambda x: [2*x[0]-x[2]-1, x[1]-3,
                       2*x[2]-x[0]+x[3]+1, x[3]+x[2]-1]
def gradientg6():
    return lambda x: [[1,2,1,1],[3,1,2,-1],[0,-1,-4,0],
                       [-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]]
def gradienth6():
    return lambda x: []

def initpoint7(n):
    x = np.zeros(n)
    x[0]=-2; x[1]=1.5; x[2]=2; x[3]=-1; x[4]=-1
    return np.array(x)
def f7():
    return lambda x: x[0]*x[1]*x[2]*x[3]*x[4]
def g7():
    return lambda x: []
def h7():
    return lambda x: [x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2-10,
                       x[1]*x[2]-5*x[3]*x[4],
                       x[0]**3+x[1]**3+1]
def gradientf7():
    return lambda x: [x[1]*x[2]*x[3]*x[4],x[0]*x[2]*x[3]*x[4],
                       x[0]*x[1]*x[3]*x[4],x[0]*x[1]*x[2]*x[4],x[0]*x[1]*x[2]*x[3]]
def gradientg7():
    return lambda x: []
def gradienth7():
    return lambda x: [[2*x[0],2*x[1],2*x[2],2*x[3],2*x[4]],
                       [0,x[2],x[1],-5*x[4],-5*x[3]],
                       [3*x[0]**2,3*x[1]**2,0,0,0]]

def initpoint8(n):
    x = np.zeros(n); x[0]=1; x[1]=5; x[2]=5; x[3]=1
    return np.array(x)
def f8():
    return lambda x: x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
def g8():
    return lambda x: [-x[0]*x[1]*x[2]*x[3]+25,
                       x[0]-5,x[1]-5,x[2]-5,x[3]-5,
                       -x[0]+1,-x[1]+1,-x[2]+1,-x[3]+1]
def h8():
    return lambda x: [x[0]**2+x[1]**2+x[2]**2+x[3]**2-40]
def gradientf8():
    return lambda x: [x[3]*(2*x[0]+x[1]+x[2]), x[0]*x[3],
                       x[0]*x[3]+1, x[0]*(x[0]+x[1]+x[2])]
def gradientg8():
    return lambda x: [[-x[1]*x[2]*x[3],-x[0]*x[2]*x[3],
                        -x[0]*x[1]*x[3],-x[0]*x[1]*x[2]],
                       [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                       [-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]]
def gradienth8():
    return lambda x: [[2*x[0],2*x[1],2*x[2],2*x[3]]]

def initpoint9(n):
    x = np.array([2,3,5,5,1,2,7,3,6,10], dtype=float)
    return x
def f9():
    return lambda x: (x[0]**2+x[1]**2+x[0]*x[1]-14*x[0]-16*x[1]
                      +(x[2]-10)**2+4*(x[3]-5)**2+(x[4]-3)**2
                      +2*(x[5]-1)**2+5*x[6]**2+7*(x[7]-11)**2
                      +2*(x[8]-10)**2+(x[9]-7)**2+45)
def g9():
    return lambda x: [
        -105+4*x[0]+5*x[1]-3*x[6]+9*x[7],
        10*x[0]-8*x[1]-17*x[6]+2*x[7],
        -8*x[0]+2*x[1]+5*x[8]-2*x[9]-12,
        3*(x[0]-2)**2+4*(x[1]-3)**2+2*x[2]**2-7*x[3]-120,
        5*x[0]**2+8*x[1]+(x[2]-6)**2-2*x[3]-40,
        0.5*(x[0]-8)**2+2*(x[1]-4)**2+3*x[4]**2-x[5]-30,
        x[0]**2+2*(x[1]-2)**2-2*x[0]*x[1]+14*x[4]-6*x[5],
        -3*x[0]+6*x[1]+12*(x[8]-8)**2-7*x[9]
    ]
def h9():
    return lambda x: []
def gradientf9():
    return lambda x: [2*x[0]+x[1]-14, 2*x[1]+x[0]-16,
                       2*(x[2]-10), 8*(x[3]-5), 2*(x[4]-3),
                       4*(x[5]-1), 10*x[6], 14*(x[7]-11),
                       4*(x[8]-10), 2*(x[9]-7)]
def gradientg9():
    return lambda x: [
        [4,5,0,0,0,0,-3,9,0,0],
        [10,-8,0,0,0,0,-17,2,0,0],
        [-8,2,0,0,0,0,0,0,5,-2],
        [6*(x[0]-2),8*(x[1]-3),4*x[2],-7,0,0,0,0,0,0],
        [10*x[0],8,2*(x[2]-6),-2,0,0,0,0,0,0],
        [x[0]-8,4*(x[1]-4),0,0,6*x[4],-1,0,0,0,0],
        [2*x[0]-2*x[1],4*(x[1]-2)-2*x[0],0,0,14,-6,0,0,0,0],
        [-3,6,0,0,0,0,0,0,24*(x[8]-8),-7]
    ]
def gradienth9():
    return lambda x: []

# ─── PROBLEMI P (per penalità sequenziali) ──────────────────────────────────
def initpointP3(n):
    x = np.zeros(n); x[0]=-2; x[1]=1
    return x
def Pf3():
    return lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
def Pg3():
    return lambda x: [-x[0]-x[1]**2,
                       -x[0]**2-x[1],
                       -x[0]**2-x[1]**2+1,
                       x[0]-0.5,
                       -x[0]-0.5]
def Ph3():
    return lambda x: []
def gradientPf3():
    return lambda x: np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),
                                  200*(x[1]-x[0]**2)])
def gradientPg3():
    return lambda x: [[-1,-2*x[1]],[-2*x[0],-1],
                       [-2*x[0],-2*x[1]],[1,0],[-1,0]]
def gradientPh3():
    return lambda x: []

def initpointP7(n):
    x = np.ones(n)
    return x
def Pf7():
    return lambda x: (x[0]-1)**2+(x[1]-2)**2+(x[2]-3)**2+(x[3]-4)**2
def Pg7():
    return lambda x: []
def Ph7():
    return lambda x: [x[0]-2, x[2]**2+x[3]**2-2]
def gradientPf7():
    return lambda x: [2*(x[0]-1),2*(x[1]-2),2*(x[2]-3),2*(x[3]-4)]
def gradientPg7():
    return lambda x: []
def gradientPh7():
    return lambda x: [[1,0,0,0],[0,0,2*x[2],2*x[3]]]

def initpointP11(n):
    x = np.array([1.0, 1.0, 0.5, 0.5])
    return x
def Pf11():
    return lambda x: -x[0]
def Pg11():
    return lambda x: [x[0]**3-x[1], x[1]-x[0]**2]
def Ph11():
    return lambda x: [x[1]-x[0]**3-x[2]**2, x[0]**2-x[1]-x[3]**2]
def gradientPf11():
    return lambda x: np.array([-1.0, 0.0, 0.0, 0.0])
def gradientPg11():
    return lambda x: [[3*x[0]**2,-1,0,0],[-2*x[0],1,0,0]]
def gradientPh11():
    return lambda x: [[-3*x[0]**2,1,-2*x[2],0],[2*x[0],-1,0,-2*x[3]]]

# ─── PENALITÀ (funzioni ausiliarie) ─────────────────────────────────────────
def Fpen(e, f, g, h):
    """Funzione obiettivo penalizzata: f(x) + (1/e)*sum(max(0,gi)^2) + (1/e)*sum(hj^2)"""
    return lambda x: (f(x)
                      + sum((1/e)*max(0, g(x)[i])**2 for i in range(len(g(x))))
                      + sum((1/e)*h(x)[j]**2 for j in range(len(h(x)))))

def gradPS(e, g, h, gradientf, gradientg, gradienth):
    """Gradiente della funzione penalizzata."""
    return lambda x: np.array([
        gradientf(x)[k]
        + sum((2/e)*gradientg(x)[i][k]*max(0, g(x)[i]) for i in range(len(g(x))))
        + sum((2/e)*gradienth(x)[j][k]*h(x)[j] for j in range(len(h(x))))
        for k in range(len(gradientf(x)))
    ])

# ─── FILLED ─────────────────────────────────────────────────────────────────
def initpointF1(n): return np.zeros(n)
def Filled1():
    return lambda x: sum((x[i]-1)**2 + 100*(x[i]**2-x[i+1])**2
                         for i in range(len(x)-1))

def initpointF4(n): return np.zeros(n)
def Filled4():
    return lambda x: ((np.exp(x[0])-x[1])**4 + 100*(x[1]-x[2])**6
                      + (np.tan(x[2]-x[3]))**4 + x[0]**8)

def initpointF5(n): return np.zeros(n)
def Filled5():
    return lambda x: (100*(x[0]**2-x[1])**2 + (x[0]-1)**2 + (x[2]-1)**2
                      + 90*(x[2]**2-x[3])**2
                      + 10.1*((x[1]-1)**2+(x[3]-1)**2)
                      + 19.8*(x[1]-1)*(x[3]-1))

def initpointF6(n): return np.zeros(n)
def Filled6():
    return lambda x: ((4-2.1*x[0]**2+(x[0]**4)/3)*x[0]**2
                      + x[0]*x[1] + (-4+4*x[1]**2)*x[1]**2)

def initpointF7(n): return np.zeros(n)
def Filled7():
    return lambda x: (
        (np.pi/len(x))*(10*(np.sin(np.pi*x[0]))**2
        + sum((x[i]-1)**2*(1+10*(np.sin(np.pi*x[i+1]))**2)
               for i in range(len(x)-1)))
        + (x[-1]-1)**2)

def initpointF8(n): return np.zeros(n)
def Filled8():
    return lambda x: (
        (1/10)*(np.sin(3*np.pi*x[0]))**2
        + sum((x[i]-1)**2*(1+10*(np.sin(3*np.pi*x[i+1]))**2)
               for i in range(len(x)-1))
        + (1/10)*(x[-1]-1)**2*(1+(np.sin(2*np.pi*x[-1]))**2))

def initpointF11(n): return np.zeros(n)
def Filled11():
    return lambda x: (
        (1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))
        *(30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2)))

def initpointF12(n): return np.zeros(n)
def Filled12():
    return lambda x: np.exp(-0.5*np.sum(x**2))

def initpointF13(n): return np.zeros(n)
def Filled13():
    return lambda x: (0.1*np.sum(np.cos(5*np.pi*np.array(x)))
                      - np.sum(np.array(x)**2))

def initpointF23(n): return np.zeros(n)
def Filled23():
    return lambda x: (
        (x[1]-(5.1*x[0]**2)/(4*np.pi**2)+(5*x[0])/np.pi-6)**2
        + 10*(1-1/(8*np.pi))*np.cos(x[0])*np.cos(x[1])*np.log(x[0]**2+x[1]**2+1)
        + 10)

def initpointF67(n): return np.zeros(n)
def Filled67():
    return lambda x: sum((2+2*i-(np.exp(i*x[0])+np.exp(i*x[1])))**2
                         for i in range(1, 11))

def initpointF95(n): return np.ones(n)
def Filled95():
    return lambda x: (1 + np.sin(x[0])**2 + np.sin(x[1])**2
                      - 0.1*np.exp(-x[0]**2-x[1]**2))

def initpointF129(n): return np.zeros(n)
def Filled129():
    return lambda x: -x[0]*x[1]*(72-2*x[0]-2*x[1])
