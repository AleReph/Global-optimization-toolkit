import numpy as np
from funzioni import (
    Fpen, gradPS,
    initpointP3,  Pf3,  Pg3,  Ph3,  gradientPf3,  gradientPg3,  gradientPh3,
    initpointP7,  Pf7,  Pg7,  Ph7,  gradientPf7,  gradientPg7,  gradientPh7,
    initpointP11, Pf11, Pg11, Ph11, gradientPf11, gradientPg11, gradientPh11,
)
from NewtonT import norma, NT


# ─── MOLTIPLICATORI KKT (stima dalla penalità) ───────────────────────────────

def _moltiplicatori(x, eps, g, h):
    """
    Stima i moltiplicatori di Lagrange dalla condizione di stazionarietà
    della funzione penalizzata:
        lambda_i = (2/eps) * max(0, g_i(x))    (disuguaglianze)
        mu_j     = (2/eps) * h_j(x)             (uguaglianze)
    """
    gx = g(x); hx = h(x)
    lam = np.array([(2.0 / eps) * max(0.0, gi) for gi in gx]) if len(gx) > 0 else np.array([])
    mu  = np.array([(2.0 / eps) * hj           for hj in hx]) if len(hx) > 0 else np.array([])
    return lam, mu


# ─── VIOLAZIONE DEI VINCOLI ──────────────────────────────────────────────────

def violazione(x, g, h):
    """Somma dei quadrati delle violazioni dei vincoli."""
    pen = sum(max(0.0, gi)**2 for gi in g(x))
    pen += sum(hj**2 for hj in h(x))
    return pen


# ─── STAZIONARIETÀ DEL LAGRANGIANO ───────────────────────────────────────────

def _grad_lagrangiano(x, gradientf, g, h, gradientg, gradienth, lam, mu):
    Lagr = np.array(gradientf(x), dtype=float)
    if len(lam) > 0:
        for i in range(len(g(x))):
            Lagr += lam[i] * np.array(gradientg(x)[i], dtype=float)
    if len(mu) > 0:
        for j in range(len(h(x))):
            Lagr += mu[j]  * np.array(gradienth(x)[j], dtype=float)
    return Lagr


# ─── VERIFICA CONDIZIONI KKT ─────────────────────────────────────────────────

def KKT(x, f, g, h, n, gradientf, gradientg, gradienth, lam, mu, tol=1e-4):
    """
    Verifica le condizioni KKT nel punto x con i moltiplicatori dati.
    Restituisce True se tutte le condizioni sono soddisfatte entro tol.
    """
    # 1. Stazionarietà del Lagrangiano
    Lagr = _grad_lagrangiano(x, gradientf, g, h, gradientg, gradienth, lam, mu)
    if norma(Lagr) > tol:
        return False

    gx = g(x); hx = h(x)

    # 2. Ammissibilità primale
    if any(gi > tol for gi in gx):
        return False
    if any(abs(hj) > tol for hj in hx):
        return False

    # 3. Dual feasibility e complementarietà
    if len(lam) > 0:
        for i in range(len(gx)):
            if lam[i] < -tol:                   # lam >= 0
                return False
            if abs(gx[i] * lam[i]) > tol:       # complementarità
                return False

    return True


# ─── STAMPA RIEPILOGO ────────────────────────────────────────────────────────

def _stampa_risultato(x, f, g, h, gradientf, gradientg, gradienth, lam, mu):
    Lagr = _grad_lagrangiano(x, gradientf, g, h, gradientg, gradienth, lam, mu)
    print(f"\n{'='*55}")
    print(f"  x*      = {np.round(x, 6).tolist()}")
    print(f"  f(x*)   = {f(x):.8f}")
    gx = g(x); hx = h(x)
    if len(gx) > 0:
        print(f"  g(x*)   = {[round(v, 6) for v in gx]}")
    if len(hx) > 0:
        print(f"  h(x*)   = {[round(v, 6) for v in hx]}")
    if len(lam) > 0:
        print(f"  lambda  = {np.round(lam, 6).tolist()}")
    if len(mu) > 0:
        print(f"  mu      = {np.round(mu, 6).tolist()}")
    print(f"  ||∇L||  = {norma(Lagr):.2e}")
    viol = violazione(x, g, h)
    print(f"  viol    = {viol:.2e}")
    print(f"{'='*55}\n")


# ─── METODO DELLE PENALITÀ SEQUENZIALI ───────────────────────────────────────

def PS(x0, g, h, f, gradientf, gradientg, gradienth, n,
       eps0=1.0, delta0=1.0, theta1=0.25, theta2=0.1, theta3=0.5,
       max_iter=300, tol_kkt=1e-4, verbose=True):
    """
    Parametri
    ----------
    x0        : punto iniziale
    g, h      : funzioni vincolo (g_i(x) <= 0,  h_j(x) = 0)
    f         : funzione obiettivo
    gradientf : gradiente di f
    gradientg : lista gradienti dei vincoli di disuguaglianza
    gradienth : lista gradienti dei vincoli di uguaglianza
    n         : dimensione
    eps0      : parametro di penalità iniziale (piccolo → penalità pesante)
    delta0    : tolleranza iniziale per il sotto-problema (NT)
    theta1    : soglia di miglioramento relativo della violazione (∈ (0,1))
    theta2    : fattore di riduzione di eps quando la violazione non migliora (∈ (0,1))
    theta3    : fattore di riduzione di delta (∈ (0,1))
    max_iter  : numero massimo di iterazioni esterne
    tol_kkt   : tolleranza per le condizioni KKT
    """
    x     = np.array(x0, dtype=float)
    eps   = float(eps0)
    delta = float(delta0)

    for k in range(max_iter):
        # ── P1: stima moltiplicatori e verifica KKT ──
        lam, mu = _moltiplicatori(x, eps, g, h)

        if KKT(x, f, g, h, n, gradientf, gradientg, gradienth, lam, mu, tol_kkt):
            if verbose:
                print(f"  → KKT soddisfatte all'iterazione {k}")
                _stampa_risultato(x, f, g, h, gradientf, gradientg, gradienth, lam, mu)
            return x

        # ── P2: minimizzazione del sotto-problema ──
        viol_old = violazione(x, g, h)

        Fp   = Fpen(eps, f, g, h)
        gFp  = gradPS(eps, g, h, gradientf, gradientg, gradienth)

        x_new = NT(x, Fp, gFp, n, delta)

        viol_new = violazione(x_new, g, h)

        # ── P3: aggiornamento parametro di penalità ──
        if viol_old < 1e-14:
            # già ammissibile: aggiorna solo delta
            pass
        elif viol_new > theta1 * viol_old:
            # violazione non migliorata abbastanza → aumenta penalità
            eps *= theta2

        # ── P4: aggiorna delta e itera ──
        delta = max(delta * theta3, 1e-9)
        x = x_new

    # Fine iterazioni – stampa comunque il risultato
    lam, mu = _moltiplicatori(x, eps, g, h)
    if verbose:
        print(f"  → Raggiunto limite di {max_iter} iterazioni")
        _stampa_risultato(x, f, g, h, gradientf, gradientg, gradienth, lam, mu)
    return x


# ─── ESECUZIONE ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np3  = 2; np7 = 4; np11 = 4

    print("=" * 60)
    print("Problema P3 — Rosenbrock con vincoli")
    print("=" * 60)
    PS(initpointP3(np3), Pg3(), Ph3(), Pf3(),
       gradientPf3(), gradientPg3(), gradientPh3(), np3)

    print("=" * 60)
    print("Problema P7 — distanza minima con uguaglianze")
    print("=" * 60)
    PS(initpointP7(np7), Pg7(), Ph7(), Pf7(),
       gradientPf7(), gradientPg7(), gradientPh7(), np7)

    print("=" * 60)
    print("Problema P11 — massimizzazione di x1 con uguaglianze")
    print("=" * 60)
    PS(initpointP11(np11), Pg11(), Ph11(), Pf11(),
       gradientPf11(), gradientPg11(), gradientPh11(), np11)
