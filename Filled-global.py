import numpy as np
from NewtonT import norma, NT
from funzioni import (
    initpointF1,   Filled1,
    initpointF4,   Filled4,
    initpointF5,   Filled5,
    initpointF6,   Filled6,
    initpointF7,   Filled7,
    initpointF8,   Filled8,
    initpointF11,  Filled11,
    initpointF12,  Filled12,
    initpointF13,  Filled13,
    initpointF23,  Filled23,
    initpointF67,  Filled67,
    initpointF95,  Filled95,
    initpointF129, Filled129,
)


# ─── GRADIENTE APPROSSIMATO (differenze finite centrali) ─────────────────────

def gradapprF(f, h=1e-5):
    """
    Gradiente di f approssimato con differenze finite centrali.
    h=1e-5 è un buon compromesso troncamento/cancellazione numerica.
    """
    def _g(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        g = np.empty(n)
        ei = np.zeros(n)
        for i in range(n):
            ei[i] = h
            g[i] = (f(x + ei) - f(x - ei)) / (2.0 * h)
            ei[i] = 0.0
        return g
    return _g


# ─── FUNZIONE FILLED ADDITIVA DI TIPO 1 ──────────────────────────────────────
#
#  la funzione filled additiva ha la forma:
#
#   U_k(x) = ψ_k(x)  +  φ_k(x)
#
# dove:
#   φ_k(x) = exp(-||x - x*_k||2 / γ2)          [collina gaussiana centrata in x*_k]
#   ψ_k(x) = τ * (min{0, f(x) - f(x*_k) + ρ})3  [penalizza la zona con f basso]
#
# La collina gaussiana rende x*_k un massimo locale della funzione filled,
# spingendo la minimizzazione locale verso un altro bacino.
# Il termine ψ_k è negativo solo dove f(x) < f(x*_k) - ρ, creando
# una depressione che attira verso minimi globali migliori.

def _filled_function(f, xstar, tau, rho, gamma):
    """
    Funzione filled additiva di Tipo 1 centrata nel minimo attuale xstar.

      U(x) = τ * (min{0, f(x) - f(xstar) + ρ})3  +  exp(-||x - xstar||2 / γ2)

    Parametri
    ----------
    tau   : peso del termine di penalità (τ > 0, più grande = penalità più forte)
    rho   : margine sul valore della funzione (ρ > 0)
    gamma : ampiezza della collina gaussiana (γ > 0)
    """
    fstar = float(f(xstar))
    gamma2 = gamma ** 2

    def U(x):
        fx   = f(x)
        diff = fx - fstar + rho
        mn   = min(0.0, diff)
        gauss = np.exp(-norma(x - xstar) ** 2 / gamma2)
        return tau * mn ** 3 + gauss

    return U


# ─── ALGORITMO DELLE FUNZIONI FILLED ─────────────────────────────────────────

def Filled(x0, f, gradientf, n, delta,
           tau=50.0, rho=0.5, gamma=0.5,
           box_lo=-10.0, box_hi=10.0,
           max_outer=50, n_restarts=8,
           rng_seed=42):
    """
    Algoritmo delle funzioni filled per ottimizzazione globale.

    Schema :
      1. Trova un minimo locale x*_k partendo da x0 con Newton Troncato.
      2. Costruisce la funzione filled U_k(x) centrata in x*_k.
      3. Minimizza U_k partendo da più punti iniziali perturbati intorno a x*_k.
      4. Se si trova y con f(y) < f(x*_k) → aggiorna x*_k = y, torna a 2.
      5. Se nessun miglioramento dopo n_restarts tentativi → Stop.

    Parametri
    ----------
    x0         : punto iniziale
    f          : funzione obiettivo
    gradientf  : gradiente di f (o gradapprF(f))
    n          : dimensione del problema
    delta      : tolleranza sul gradiente per NT
    tau        : peso del termine ψ nella funzione filled (τ)
    rho        : margine sulla funzione (ρ)
    gamma      : ampiezza della collina gaussiana (γ) — es. {0.1, 0.5, 1.0}
    box_lo     : bound inferiore della box feasible
    box_hi     : bound superiore della box feasible
    max_outer  : numero massimo di iterazioni esterne
    n_restarts : numero di punti di partenza perturbati per ogni iterazione esterna
    rng_seed   : seme per riproducibilità (None = casuale)
    """
    rng   = np.random.default_rng(rng_seed)
    x0    = np.asarray(x0, dtype=float)
    diam  = box_hi - box_lo

    # ── Passo 1: minimo locale iniziale ──
    xstar  = NT(x0, f, gradientf, n, delta)
    xstar  = np.clip(xstar, box_lo, box_hi)
    fstar  = f(xstar)

    for outer in range(max_outer):
        # ── Passo 2: costruisce la funzione filled centrata in xstar ──
        uf  = _filled_function(f, xstar, tau, rho, gamma)
        guf = gradapprF(uf)                         # gradiente approssimato di U

        trovato    = False
        best_x_new = xstar.copy()
        best_f_new = fstar

        # ── Passo 3: minimizza U da più punti di partenza ──
        # Strategia di perturbazione:
        #   - scale geometriche (10^0, 10^{-1}, ...) × direzioni cardinali ± e_i
        #   - punti casuali nella box
        start_points = []

        # perturbazioni scalate lungo gli assi e le diagonali
        for exp_i in range(4):                          # scale 1, 0.1, 0.01, 0.001
            step_size = diam * 10.0 ** (-exp_i)
            for i in range(n):
                ei = np.zeros(n); ei[i] = 1.0
                for sign in (+1.0, -1.0):
                    p = xstar + sign * step_size * ei
                    start_points.append(np.clip(p, box_lo, box_hi))

        # punti casuali nella box
        for _ in range(n_restarts):
            p = rng.uniform(box_lo, box_hi, size=n)
            start_points.append(p)

        for xs in start_points:
            try:
                z = NT(xs, uf, guf, n, delta * 10)   # tolleranza rilassata per la filled
                z = np.clip(z, box_lo, box_hi)
            except Exception:
                continue

            # accetta se: diverso da xstar  e  migliora f
            if (not np.allclose(z, xstar, atol=1e-4) and f(z) < best_f_new - 1e-8):
                best_f_new = f(z)
                best_x_new = z.copy()
                trovato    = True

        if not trovato:
            # nessun miglioramento → stop
            break

        # ── Passo 4: aggiorna il minimo corrente ──
        # raffiniamo il nuovo punto con NT sulla funzione originale
        xstar = NT(best_x_new, f, gradientf, n, delta)
        xstar = np.clip(xstar, box_lo, box_hi)
        fstar = f(xstar)

    return xstar, fstar


# ─── ESECUZIONE ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Tabella dei problemi:
    #   (nome, n, init_fn, func_fn, delta, gamma, (box_lo, box_hi), tau, rho)
    problemi = [
        ("F1",   4, initpointF1,   Filled1,   1e-8, 0.5,  (-2,   2),   50.0, 0.5),
        ("F4",   4, initpointF4,   Filled4,   1e-8, 0.5,  (-2,   2),   50.0, 0.5),
        ("F5",   4, initpointF5,   Filled5,   1e-8, 0.5,  (-2,   2),   50.0, 0.5),
        ("F6",   2, initpointF6,   Filled6,   1e-8, 0.5,  (-3,   3),   20.0, 0.3),
        ("F7",   5, initpointF7,   Filled7,   1e-6, 0.5,  (-2,   2),   50.0, 0.5),
        ("F8",   5, initpointF8,   Filled8,   1e-8, 0.5,  (-2,   2),   50.0, 0.5),
        ("F11",  2, initpointF11,  Filled11,  1e-8, 0.5,  (-2,   2),   50.0, 0.5),
        ("F12",  4, initpointF12,  Filled12,  1e-8, 0.5,  (-2,   2),   10.0, 0.1),
        ("F13",  4, initpointF13,  Filled13,  1e-8, 0.5,  (-1,   1),   10.0, 0.1),
        ("F23",  2, initpointF23,  Filled23,  1e-8, 1.0,  (-5,  15),   50.0, 0.5),
        ("F67",  2, initpointF67,  Filled67,  1e-8, 0.05, (-1,   1),   20.0, 0.3),
        ("F95",  2, initpointF95,  Filled95,  1e-8, 0.5,  (-3,   3),   10.0, 0.1),
        ("F129", 2, initpointF129, Filled129, 1e-8, 2.0,  (0,   36),   20.0, 1.0),
    ]

    print(f"{'Prob':<6} {'f*':>14}  {'x*'}")
    print("-" * 70)
    for nome, n, initpt, func, delta, gamma, (blo, bhi), tau, rho in problemi:
        x0 = initpt(n)
        f  = func()
        gf = gradapprF(f)
        sol, fval = Filled(
            x0, f, gf, n, delta,
            tau=tau, rho=rho, gamma=gamma,
            box_lo=blo, box_hi=bhi,
        )
        print(f"{nome:<6} {fval:>14.6f}  {np.round(sol, 5).tolist()}")
