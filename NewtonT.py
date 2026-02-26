import numpy as np

# ─── UTILITÀ ────────────────────────────────────────────────────────────────

def norma(v):
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.dot(v, v)))


# ─── LINE SEARCH DI ARMIJO ──────────────────────────────────────────────────

def linesearch_armijo(x, f, d, gd, gamma=1e-4, beta=0.5, max_iter=50):
    """
    Backtracking line search (Armijo).
    Restituisce alfa t.c. f(x + alfa*d) <= f(x) + alfa*gamma*gd.
    """
    alfa = 1.0
    fx = f(x)
    for _ in range(max_iter):
        if f(x + alfa * d) <= fx + alfa * gamma * gd:
            return alfa
        alfa *= beta
    return alfa


# ─── PRODOTTO HESSIANA × VETTORE (differenze finite centrali) ───────────────

def hess_vec(x, v, grad, h=1e-5):
    """
    Approssima H(x)*v con differenze finite centrali sul gradiente.
    Più stabile rispetto all'uso diretto di f.
    """
    v = np.asarray(v, dtype=float)
    nv = norma(v)
    if nv < 1e-14:
        return np.zeros_like(x)
    v_hat = v / nv
    return (np.asarray(grad(x + h * v_hat)) - np.asarray(grad(x - h * v_hat))) / (2 * h) * nv


# ─── GRADIENTE CONIUGATO TRONCATO (GCT / Steihaug-CG) ──────────────────────

def GCT(x, grad, delta_tr=None, eps_rel=0.1, max_cg=None):
    """
    Gradiente coniugato troncato per il sotto-problema di Newton.

    Risolve (approssimativamente) H*d = -g con:
    - troncamento quando la curvatura è non-positiva (direzione di discesa garantita)
    - troncamento quando la norma del residuo è sufficientemente piccola
    - opzionalmente: troncamento alla frontiera del trust-region (delta_tr)

    Restituisce una direzione di discesa d.
    """
    gx = np.asarray(grad(x), dtype=float)
    n = len(gx)
    if max_cg is None:
        max_cg = max(n, 30)

    p = np.zeros(n)           # soluzione corrente (partenza in 0)
    r = -gx.copy()            # residuo iniziale = -g (perché Hp = 0 all'inizio)
    s = r.copy()              # direzione di ricerca coniugata

    rr = np.dot(r, r)
    rr0 = rr
    tol = eps_rel * np.sqrt(rr0)

    if np.sqrt(rr0) < 1e-14:
        return p

    for _ in range(max_cg):
        Hs = hess_vec(x, s, grad)
        sHs = np.dot(s, Hs)

        # Curvatura negativa o nulla → direzione di discesa pura
        if sHs <= 1e-10 * np.dot(s, s):
            # se p è già diverso da 0, restituiamo p (già una buona direzione)
            # altrimenti usiamo -g
            if norma(p) < 1e-14:
                return -gx
            return p

        alfa = rr / sHs
        p_new = p + alfa * s

        # Check trust-region se richiesto
        if delta_tr is not None and norma(p_new) >= delta_tr:
            # tronca sulla frontiera della trust-region
            # risolvi ||p + sigma*s||^2 = delta_tr^2
            pp = np.dot(p, p)
            ps = np.dot(p, s)
            ss = np.dot(s, s)
            disc = ps**2 - ss * (pp - delta_tr**2)
            if disc >= 0:
                sigma = (-ps + np.sqrt(max(0.0, disc))) / ss
                return p + sigma * s
            return p

        p = p_new
        r = r - alfa * Hs
        rr_new = np.dot(r, r)

        # Criterio di arresto: residuo sufficientemente piccolo
        if np.sqrt(rr_new) <= tol:
            return p

        beta = rr_new / rr
        s = r + beta * s
        rr = rr_new

    return p


# ─── NEWTON TRONCATO ────────────────────────────────────────────────────────

def NT(x0, f, grad, n, delta, max_iter=2000, gamma=1e-4):
    """
    Metodo di Newton Troncato con line search di Armijo.

    Parametri
    ----------
    x0     : punto iniziale (array-like)
    f      : funzione obiettivo  f: R^n -> R
    grad   : gradiente di f      grad: R^n -> R^n
    n      : dimensione del problema
    delta  : tolleranza sul gradiente (criterio di arresto: ||grad(x)|| <= delta)
    max_iter: numero massimo di iterazioni
    gamma  : parametro Armijo
    """
    x = np.array(x0, dtype=float)

    for k in range(max_iter):
        gx = np.asarray(grad(x), dtype=float)
        ng = norma(gx)

        if ng <= delta:
            break

        # Direzione di Newton troncata
        d = GCT(x, grad, eps_rel=min(0.5, np.sqrt(ng)))

        gd = np.dot(gx, d)

        # Se la direzione non è di discesa, usa l'anti-gradiente
        if gd >= -1e-14 * ng * norma(d):
            d = -gx
            gd = -ng**2

        alfa = linesearch_armijo(x, f, d, gd, gamma=gamma)

        step = alfa * d
        if norma(step) < 1e-12 * (1.0 + norma(x)):
            break

        x = x + step

    return x


# ─── TEST RAPIDO ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from funzioni import initpointprova, fprova, grad_fprova
    print("Test Newton Troncato su Rastrigin 2D:")
    x0 = initpointprova(2)
    sol = NT(x0, fprova(), grad_fprova(), 2, 1e-6)
    print(f"  Soluzione: {sol}")
    print(f"  f(sol)   : {fprova()(sol):.6f}")
