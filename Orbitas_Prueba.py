"""
╔══════════════════════════════════════════════════════════════════╗
║        SIMULADOR DE ÓRBITAS CLÁSICAS — Interfaz Interactiva      ║
║               FIS 3003 Mecánica Clásica  · RK4                   ║
║                         Profesor: Carlos Marin                   ║
║                 Autor: Ezequiel Guerrero                         ║
╚══════════════════════════════════════════════════════════════════╝
  La logica del programa permite ingresar dos puntos de borde (r₁,θ₁) y (r₂,θ₂) en la interfaz.
  El programa calcula automáticamente la trayectoria y la anima usando matplotlyb.
  Física utilizada en el programa (se explica todo procedimiento matematico en el LaTex adjunto):
    r̈ = l²/r³ − GM/r²       (EDO radial)
    θ̇ = l/r²                 (conservación de L)
    r(θ) = p/(1 + ε·cos θ)   (ecuación de la cónica)
Se implemento tambien una funcion para el caso del colapso usando una fuerza disipativa.
  Modo Espiral de Colapso:
    Añade disipación lineal: F_drag = −b·m·v
    → ṙ = vr,  v̇r = l²/r³ − GM/r² − b·vr,  θ̇ = l/r²,  l̇ = −b·l
    La partícula gira y se acerca hasta colapsar en N vueltas.
Uso (En la terminal de python solo se debe escribir esto para que funcione el programa):
    python Orbitas.py
Una ves dentro del programa en la parte izquierda se tiene la opcion de cambiar la masa central
ya sea por la del Sol o la de la Tierra,tambien se puede meter las condiciones de borde (r en UA)
y hay un boton para simular el caso de choque. Una vez acabada la simulacion se puede repetir la misma
,ingresar nuevo valores para realizar otra simulacion o finalizar el programa.
"""
# ── Dependencias ────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
# ═══════════════════════════════════════════════════════════════════
#  CONSTANTES FÍSICAS
# ═══════════════════════════════════════════════════════════════════
G        = 6.674e-11   #m³ kg⁻¹ s⁻²
M_SOL    = 1.989e30   #kg
M_TIERRA = 5.972e24   #kg
R_TIERRA = 6.371e6    #m(radio de la Tierra)
R_SOL    = 6.957e8    #m(radio del Sol)
UA       = 1.496e11   #m(Unidad Astronómica)
DIA      = 86400.0    #s
# ═══════════════════════════════════════════════════════════════════
#  FÍSICA — órbitas conservativas (sin disipación)
# ═══════════════════════════════════════════════════════════════════
def calcular_e_l(r1, th1, r2, th2, GM):
    """
    Calcula e, l, ε y p a partir de dos puntos de la órbita.
    Ecuación de la cónica evaluada en ambos puntos:
        r₁(1 + ε·cos θ₁) = p   ...(I)
        r₂(1 + ε·cos θ₂) = p   ...(II)
    Despejando:
        ε = (r₂ − r₁) / (r₁·cos θ₁ − r₂·cos θ₂)
        p = r₁(1 + ε·cos θ₁)
        l = √(p·GM)
        e = (ε²−1)·GM² / (2l²)
    Aqui se verifican que los puntos de borde tengan sentido fisico,por ejemplo el de semilatuz rectum
    significa que el valor de la distancia del foco hasta la curva medida perpendicularmente al eje focal
    es negavita,es decir que no representa a una conica conocida del caso circular o eliptica.
    """
    c1, c2 = np.cos(th1), np.cos(th2)
    denom  = r1 * c1 - r2 * c2
    if abs(denom) < 1e-30:
        raise ValueError(
            "Puntos degenerados: elige ángulos más distintos entre sí.")
    eps = (r2 - r1) / denom
    p   = r1 * (1.0 + eps * c1)
    if p <= 0:
        raise ValueError(
            f"Semi-latus rectum p ≤ 0. Revisa los puntos de borde.")
    l = np.sqrt(p * GM)
    e = (eps**2 - 1.0) * GM**2 / (2.0 * l**2)
    if   abs(eps) < 1e-5:        tipo = "Circular    (ε ≈ 0)"
    elif eps < 1.0 - 1e-5:      tipo = "Elíptica    (ε < 1)"
    elif abs(eps - 1.0) < 3e-4: tipo = "Parabólica  (ε ≈ 1)"
    else:                        tipo = "Hiperbólica (ε > 1)"
    return e, l, eps, p, tipo
def condiciones_iniciales(r1, th1, e, l, GM, eps):
    """
    vr₀ desde conservación de energía; signo desde dr/dθ en (r₁,θ₁).
        e = vr²/2 + (l/r)²/2 − GM/r
        es decir que vr = ±√(2e + 2GM/r₁ − l²/r₁²)
    Signo: si dr/dθ < 0 en θ₁ entonces la partícula se acerca,entonces vr < 0.
    """
    p   = l**2 / GM
    vr2 = max(0.0, 2.0 * e + 2.0 * GM / r1 - (l / r1)**2)
    vr  = np.sqrt(vr2)
    den = 1.0 + eps * np.cos(th1)
    if abs(den) > 1e-10:
        dr_dth = p * eps * np.sin(th1) / den**2
        if dr_dth < 0.0:
            vr = -vr
    omega = l / r1**2
    return float(vr), float(omega)
def derivadas(s, GM, l):
    """
    ds/dt donde s = [r, vr, θ].
    ṡ = [vr,  l²/r³ − GM/r²,  l/r²]
    """
    r, vr, th = s
    if r < 1.0e3:
        return np.zeros(3)
    return np.array([vr, l**2 / r**3 - GM / r**2, l / r**2])
def rk4_paso(s, dt, GM, l):
    """Un paso Runge-Kutta de 4° orden (caso conservativo)."""
    k1 = derivadas(s,             GM, l)
    k2 = derivadas(s + dt/2*k1,  GM, l)
    k3 = derivadas(s + dt/2*k2,  GM, l)
    k4 = derivadas(s + dt*k3,    GM, l)
    return s + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
def integrar(r0, vr0, th0, GM, l, t_max, dt,
             r_escape=None, r_impacto=None):
    """
    Integra con RK4 desde t=0 hasta t=t_max.
    Paradas: colapso (r<1km) | escape (r>r_escape) | impacto (r<r_impacto).
    Retorna: t, r, th, vr, x, y, estado
    """
    N      = int(t_max / dt) + 2
    t_buf  = np.empty(N); r_buf  = np.empty(N)
    th_buf = np.empty(N); vr_buf = np.empty(N)
    s      = np.array([r0, vr0, th0], dtype=float)
    n_fin  = N
    estado = 'normal'
    for i in range(N):
        t_buf[i]  = i * dt;  r_buf[i]  = s[0]
        th_buf[i] = s[2];    vr_buf[i] = s[1]
        if s[0] < 1.0e3:
            n_fin = i + 1; estado = 'colapso'; break
        if r_impacto is not None and s[0] <= r_impacto and i > 0:
            n_fin = i + 1; estado = 'impacto'; break
        if r_escape is not None and s[0] > r_escape and i > 5:
            n_fin = i + 1; estado = 'escape';  break
        s = rk4_paso(s, dt, GM, l)
    t  = t_buf[:n_fin];  r  = r_buf[:n_fin]
    th = th_buf[:n_fin]; vr = vr_buf[:n_fin]
    x  = r * np.cos(th); y  = r * np.sin(th)
    return t, r, th, vr, x, y, estado
def energia_especifica(r, vr, l, GM):
    """e = vr²/2 + (l/r)²/2 − GM/r"""
    return 0.5 * vr**2 + 0.5 * (l / r)**2 - GM / r
"""def elegir_parametros_integracion(eps, p, GM, R_central, es_tierra):
    """
"""Elige automáticamente dt y t_max según el tipo de órbita.
    Retorna: t_max, dt, r_escape, r_impacto"""
"""
    r_impacto = None
    if eps < 1.0:
        a     = p / (1.0 - eps**2)
        T     = 2.0 * np.pi * np.sqrt(a**3 / GM)
        t_max = T * 1.15
        dt    = T / 60_000
        r_esc = None
        if R_central > 0 and p / (1.0 + eps) < R_central:
            r_impacto = R_central
    elif abs(eps - 1.0) < 3e-4:
        v_p   = np.sqrt(2.0 * GM / (p / 2.0))
        t_max = 6.0 * p / v_p
        dt    = t_max / 80_000
        r_esc = 8.0 * p
    else:
        v_inf = np.sqrt(2.0 * abs(GM * (eps**2 - 1.0)) / p * (eps**2 - 1))
        t_max = 4.0 * p / np.sqrt(GM / p)
        dt    = t_max / 80_000
        r_esc = 8.0 * p
        if R_central > 0 and p / (1.0 + eps) < R_central:
            r_impacto = R_central
    if es_tierra:
        t_max = min(t_max, 86_400.0 * 5)
        dt    = min(dt, 2.0)
    dt = np.clip(dt, 0.01, 8.0 * 3600.0)
    return t_max, dt, r_esc if eps >= 1.0 - 1e-5 else None, r_impacto"""
def elegir_parametros_integracion(eps, p, GM, R_central, es_tierra):
    r_impacto = None
    if eps < 1.0:
        a     = p / (1.0 - eps**2)
        T     = 2.0 * np.pi * np.sqrt(a**3 / GM)
        t_max = T * 1.15
        dt    = T / 60_000
        r_esc = None
        if R_central > 0 and p / (1.0 + eps) < R_central:
            r_impacto = R_central
    elif abs(eps - 1.0) < 3e-4:
        v_p   = np.sqrt(2.0 * GM / (p / 2.0))
        t_max = 8.0 * p / v_p  
        dt    = t_max / 80_000
        r_esc = 10.0 * p     
    else:
        v_inf = np.sqrt(GM * (eps**2 - 1.0) / p) 
        r_esc = 10.0 * p
        t_max = r_esc / v_inf * 1.2   
        dt    = t_max / 80_000
        if R_central > 0 and p / (1.0 + eps) < R_central:
            r_impacto = R_central
    if es_tierra:
        t_max = min(t_max, 86_400.0 * 5)
        dt    = min(dt, 2.0)
    dt = np.clip(dt, 0.01, 8.0 * 3600.0)
    return t_max, dt, r_esc if eps >= 1.0 else None, r_impacto
# ═══════════════════════════════════════════════════════════════════
#  FÍSICA — espiral de colapso (con disipación)
# ═══════════════════════════════════════════════════════════════════
def calcular_b_disipacion(r0, GM, N_orbits, r_min=1.0e3):
    """
    Coeficiente de disipación b tal que la órbita colapsa en N vueltas
    períodos orbitales iniciales.
    Derivación (órbita circular límite):
        l(t)=l₀exp(−bt),entonces  r_circ(t)≈r₀exp(−2bt)
        Tiempo de colapso:t_c=ln(r₀/r_min)/(2b)=NT₀
        Entonces  b=ln(r₀/r_min)/(2NT₀)
    """
    T0     = 2.0 * np.pi * np.sqrt(r0**3 / GM)
    ln_fac = np.log(max(r0 / max(r_min, 1.0), 2.0))
    b      = ln_fac / (2.0 * N_orbits * T0)
    return b, T0
def derivadas_espiral(s, GM, b):
    """
    Sistema con disipación lineal: F_drag=−b·m·v.
    Estado 4D:s=[r,vr,θ,l] (l ya NO es constante)
    Componentes del drag en polares:
      - Radial:a_r=−b·vr
      - Tangencial:τ/m=−b·r·ω ,entonces se define  l̇ =−b·l
    Ecuaciones finales:
        ṙ=vr
        v̇r=l²/r³−GM/r²−b·vr
        θ̇ =l/r²
        l̇=−b·l
    """
    r, vr, th, l = s
    if r < 1.0e3:
        return np.zeros(4)
    r2 = r * r
    return np.array([
        vr,
        l * l / (r2 * r) - GM / r2 - b * vr,
        l / r2,
        -b * l,
    ])
def rk4_paso_espiral(s, dt, GM, b):
    """Un paso RK4 para el sistema disipativo (estado 4D)."""
    k1 = derivadas_espiral(s,             GM, b)
    k2 = derivadas_espiral(s + dt/2*k1,  GM, b)
    k3 = derivadas_espiral(s + dt/2*k2,  GM, b)
    k4 = derivadas_espiral(s + dt*k3,    GM, b)
    return s + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
def integrar_espiral(r0, vr0, th0, l0, GM, b, t_max, dt,
                     R_central=0.0, r_colapso=1.0e4):
    """
    Integra las EDOs con disipación desde t=0 hasta colapso o t_max.
    Condiciones de parada:
      -r<r_colapso netonces se genera en la pantalla  'colapso'
      -r≤R_central entocnes se genera en la simulacion  'impacto' (choque con la superficie)
    Retorna: t,r,th,vr,ls,x,y,estado
    """
    N      = int(t_max / dt) + 2
    t_buf  = np.empty(N); r_buf  = np.empty(N)
    th_buf = np.empty(N); vr_buf = np.empty(N)
    l_buf  = np.empty(N)
    s      = np.array([r0, vr0, th0, l0], dtype=float)
    n_fin  = N
    estado = 'normal'
    for i in range(N):
        t_buf[i]  = i * dt
        r_buf[i]  = s[0]
        th_buf[i] = s[2]
        vr_buf[i] = s[1]
        l_buf[i]  = s[3]
        if s[0] < r_colapso:
            n_fin = i + 1; estado = 'colapso'; break
        if R_central > 0 and s[0] <= R_central and i > 0:
            n_fin = i + 1; estado = 'impacto'; break
        s = rk4_paso_espiral(s, dt, GM, b)
    sl = slice(None, n_fin)
    t  = t_buf[sl];  r  = r_buf[sl]
    th = th_buf[sl]; vr = vr_buf[sl]
    ls = l_buf[sl]
    x  = r * np.cos(th);  y = r * np.sin(th)
    return t, r, th, vr, ls, x, y, estado
# ═══════════════════════════════════════════════════════════════════
#  GUI — Clase principal
# ═══════════════════════════════════════════════════════════════════
"Aqui se programa la parte de la simulacion ya realizado una ves todo el proceso matematico,se usa colores oscuros porque se simula el espacio."
# Paleta de colores
C = dict(
    bg        = '#080c0a',
    panel     = '#0a100d',
    border    = '#1a3a20',
    green     = '#00ff88',
    green_dim = '#005533',
    green_mid = '#00cc66',
    blue      = '#44aaff',
    red       = '#ff4466',
    text      = '#c8e8d0',
    dim       = '#4a7a5a',
    entry_bg  = '#050f08',
    orbit     = '#00ff88',
    r_plot    = '#cc88ff',
    th_plot   = '#88ffcc',
    e_plot    = '#ffaa44',
    l_plot    = '#44ccff',
    # colores espiral
    orange    = '#ff9944',
    orange_dk = '#552200',
)
FONT_MONO=('Courier New',10)
FONT_MONO_S=('Courier New',8)
FONT_MONO_L=('Courier New',13,'bold')
FONT_MONO_T=('Courier New',11,'bold')
class OrbitalSimulator:
    """
    Interfaz gráfica para simulación de órbitas clásicas.
    Modos: Estándar (conservativo) y Espiral de Colapso (disipativo).
    Autor: Ezequiel Guerrero
    """
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.anim   = None
        self.fig    = None
        self.canvas = None
        self.current_params = None
        self._sim_done = False
        self._configure_window()
        self._build_panels()
        self._build_input_ui()
    # ── Ventana ─────────────────────────────────────────────────────
    def _configure_window(self):
        self.root.title(
            "⬡  SIMULADOR DE ÓRBITAS CLÁSICAS  |  Ezequiel Guerrero  |  Mecánica Clásica")
        self.root.configure(bg=C['bg'])
        self.root.geometry("1420x820")
        self.root.minsize(1100, 650)
        self.root.resizable(True, True)
    # ── Layout ──────────────────────────────────────────────────────
    def _build_panels(self):
        self.left = tk.Frame(self.root, bg=C['bg'], width=420)
        self.left.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 6), pady=12)
        self.left.pack_propagate(False)
        tk.Frame(self.root, bg=C['border'], width=1).pack(
            side=tk.LEFT, fill=tk.Y, pady=20)
        self.right = tk.Frame(self.root, bg=C['bg'])
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH,
                        expand=True, padx=(6, 12), pady=12)
    # ══════════════════════════════════════════════════════════════
    #  PANEL DE ENTRADA
    # ══════════════════════════════════════════════════════════════
    def _build_input_ui(self):
        for w in self.left.winfo_children():
            w.destroy()
        # ── Cabecera ──────────────────────────────────────────────
        tk.Label(self.left, text="◈ Simulador Orbitas Mecanica Clasica",
                 font=FONT_MONO_L, bg=C['bg'], fg=C['green']).pack(
                     pady=(14, 2))
        tk.Label(self.left, text="Mecánica Clásica",
                 font=FONT_MONO_S, bg=C['bg'], fg=C['dim']).pack(pady=(0, 1))
        tk.Label(self.left, text="Ezequiel Guerrero",
                 font=('Courier New', 9, 'italic'),
                 bg=C['bg'], fg='#5a8a6a').pack(pady=(0, 8))
        # ── Log / terminal ────────────────────────────────────────
        log_outer = tk.Frame(self.left, bg=C['green_dim'], bd=1)
        log_outer.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.log = tk.Text(
            log_outer, height=6,
            font=('Courier New', 8),
            bg='#04080a', fg='#00cc66',
            insertbackground=C['green'],
            state=tk.DISABLED, wrap=tk.WORD,
            relief=tk.FLAT, padx=8, pady=6,
            selectbackground='#004422')
        sb = tk.Scrollbar(log_outer, command=self.log.yview,
                          bg=C['bg'], troughcolor=C['bg'])
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.pack(fill=tk.X)
        self._log("╔══ Sistema iniciado ══╗")
        self._log("║ r en UA (Sol) o R⊕ (Tierra).")
        self._log("║ θ en grados.")
        self._log("╚══════════════════════╝")
        # ── Cuerpo central ────────────────────────────────────────
        self._sep("CUERPO CENTRAL")
        self.body_var = tk.StringVar(value='Sol')
        body_row = tk.Frame(self.left, bg=C['bg'])
        body_row.pack(fill=tk.X, padx=10, pady=4)
        for val, lbl, fg in [
            ('Sol',    '☀  Sol   (M_☉, r en UA)',   '#ffdd55'),
            ('Tierra', '🌍  Tierra (M_⊕, r en R⊕)', '#55ccff'),
        ]:
            tk.Radiobutton(
                body_row, text=lbl, variable=self.body_var, value=val,
                font=FONT_MONO_S, bg=C['bg'], fg=fg,
                selectcolor=C['entry_bg'],
                activebackground=C['bg'], activeforeground=fg
            ).pack(side=tk.LEFT, padx=8)
        # ── Condición de borde 1 ──────────────────────────────────
        self._sep("CONDICIÓN DE BORDE 1  (r₁, θ₁)")
        self.e_r1  = self._field("r₁", "UA  o  R⊕")
        self.e_th1 = self._field("θ₁", "grados")
        # ── Condición de borde 2 ──────────────────────────────────
        self._sep("CONDICIÓN DE BORDE 2  (r₂, θ₂)")
        self.e_r2  = self._field("r₂", "UA  o  R⊕")
        self.e_th2 = self._field("θ₂", "grados")
        # ── Modo de simulación ────────────────────────────────────
        self._sep("MODO DE SIMULACIÓN")
        self.modo_var = tk.StringVar(value='normal')
        modo_row = tk.Frame(self.left, bg=C['bg'])
        modo_row.pack(fill=tk.X, padx=10, pady=(4, 2))
        # Frame contenedor de opciones de espiral (siempre en el layout,
        # pero su hijo interno se muestra/oculta sin romper el orden de pack)
        self.espiral_frame = tk.Frame(self.left, bg=C['bg'])
        self.espiral_frame.pack(fill=tk.X, padx=10)
        # Fila interior de la opción espiral
        esp_row = tk.Frame(self.espiral_frame, bg=C['bg'])
        # Entry de número de vueltas
        tk.Label(esp_row, text=" N =", font=FONT_MONO,
                 bg=C['bg'], fg=C['orange'], width=6, anchor='e').pack(side=tk.LEFT)
        self.e_nvueltas = tk.Entry(
            esp_row, font=FONT_MONO, width=7,
            bg=C['entry_bg'], fg=C['orange'],
            insertbackground=C['orange'],
            relief=tk.FLAT, highlightthickness=1,
            highlightbackground=C['orange_dk'],
            highlightcolor=C['orange'])
        self.e_nvueltas.insert(0, "5")
        self.e_nvueltas.pack(side=tk.LEFT, padx=5)
        tk.Label(esp_row, text="vueltas al colapso",
                 font=('Courier New', 8), bg=C['bg'], fg=C['dim']).pack(side=tk.LEFT)
        self.e_nvueltas.bind('<Return>', lambda _: self._on_simulate())
        def _toggle_modo(*_):
            if self.modo_var.get() == 'espiral':
                esp_row.pack(fill=tk.X, pady=3)
                self._log("Modo: Espiral de Colapso.")
                self._log("La órbita decaerá en ~N vueltas.")
            else:
                esp_row.pack_forget()
        for val, lbl, fg in [
            ('normal',  '◉  Estándar',            C['green']),
            ('espiral', '🌀  Espiral de Colapso',  C['orange']),
        ]:
            tk.Radiobutton(
                modo_row, text=lbl, variable=self.modo_var, value=val,
                font=FONT_MONO_S, bg=C['bg'], fg=fg,
                selectcolor=C['entry_bg'],
                activebackground=C['bg'], activeforeground=fg,
                command=_toggle_modo
            ).pack(side=tk.LEFT, padx=8)
        # ── Botón simular ─────────────────────────────────────────
        tk.Frame(self.left, bg=C['border'], height=1).pack(
            fill=tk.X, padx=10, pady=(12, 0))
        self.btn_sim = tk.Button(
            self.left, text="▶  SIMULAR",
            font=FONT_MONO_T,
            bg=C['green_mid'], fg='#000000',
            activebackground=C['green'],
            activeforeground='#000000',
            relief=tk.FLAT, cursor='hand2',
            padx=20, pady=11,
            command=self._on_simulate)
        self.btn_sim.pack(fill=tk.X, padx=10, pady=10)
        self.post = tk.Frame(self.left, bg=C['bg'])
        self.post.pack(fill=tk.X, padx=10)
        # Pie de página
        tk.Label(self.left,
                 text="Ezequiel Guerrero  ·  Mecánica Clásica",
                 font=('Courier New', 7, 'italic'),
                 bg=C['bg'], fg='#3a5a4a'
                 ).pack(side=tk.BOTTOM, pady=(0, 2))
        tk.Label(self.left,
                 text="F = −GMm/r²  |  ε = √(1 + 2l²e/G²M²)",
                 font=('Courier New', 7), bg=C['bg'], fg=C['dim']
                 ).pack(side=tk.BOTTOM, pady=(0, 0))
    def _sep(self, label: str):
        tk.Label(self.left, text=label,
                 font=('Courier New', 8, 'bold'),
                 bg=C['bg'], fg=C['dim'], anchor='w'
                 ).pack(fill=tk.X, padx=10, pady=(10, 0))
        tk.Frame(self.left, bg=C['border'], height=1
                 ).pack(fill=tk.X, padx=10, pady=(2, 4))
    def _field(self, label: str, hint: str) -> tk.Entry:
        row = tk.Frame(self.left, bg=C['bg'])
        row.pack(fill=tk.X, padx=10, pady=3)
        tk.Label(row, text=f" {label} =", font=FONT_MONO,
                 bg=C['bg'], fg=C['green'], width=6, anchor='e'
                 ).pack(side=tk.LEFT)
        e = tk.Entry(row, font=FONT_MONO,
                     bg=C['entry_bg'], fg=C['green'],
                     insertbackground=C['green'],
                     relief=tk.FLAT,
                     highlightthickness=1,
                     highlightbackground=C['green_dim'],
                     highlightcolor=C['green_mid'])
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Label(row, text=hint, font=('Courier New', 8),
                 bg=C['bg'], fg=C['dim'], width=12, anchor='w'
                 ).pack(side=tk.LEFT)
        e.bind('<Return>', lambda _: self._on_simulate())
        return e
    def _log(self, msg: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, f"  {msg}\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)
    # ══════════════════════════════════════════════════════════════
    #  BOTONES POST-SIMULACIÓN
    # ══════════════════════════════════════════════════════════════
    def _show_post_buttons(self):
        for w in self.post.winfo_children():
            w.destroy()
        tk.Frame(self.post, bg=C['border'], height=1
                 ).pack(fill=tk.X, pady=(10, 6))
        tk.Label(self.post, text="── Simulación completada ──",
                 font=FONT_MONO_S, bg=C['bg'], fg=C['dim']
                 ).pack(pady=(0, 8))
        tk.Button(self.post, text="↻   REPETIR SIMULACIÓN",
                  font=FONT_MONO_T,
                  bg='#0a2a15', fg=C['green'],
                  activebackground='#0f3a1f',
                  activeforeground=C['green'],
                  relief=tk.FLAT, cursor='hand2', pady=9,
                  command=self._on_repeat
                  ).pack(fill=tk.X, pady=(0, 4))
        tk.Button(self.post, text="◈   NUEVOS VALORES",
                  font=FONT_MONO_T,
                  bg='#0a1a2a', fg=C['blue'],
                  activebackground='#0f2a3f',
                  activeforeground=C['blue'],
                  relief=tk.FLAT, cursor='hand2', pady=9,
                  command=self._on_new_values
                  ).pack(fill=tk.X, pady=(0, 4))
        tk.Button(self.post, text="✕   SALIR",
                  font=FONT_MONO_T,
                  bg='#2a0a15', fg=C['red'],
                  activebackground='#3f0f1f',
                  activeforeground=C['red'],
                  relief=tk.FLAT, cursor='hand2', pady=9,
                  command=self._on_exit
                  ).pack(fill=tk.X)
        self._log("─" * 30)
        self._log("Simulación finalizada.")
        self._log("Elige una opción abajo.")
    def _on_repeat(self):
        if self.current_params is None:
            return
        self._stop_anim()
        self._log("─" * 30)
        self._log("Repitiendo simulación...")
        for w in self.post.winfo_children():
            w.destroy()
        self._sim_done = False
        self._run_with_params(self.current_params)
    def _on_new_values(self):
        self._stop_anim()
        self._clear_canvas()
        self.current_params = None
        self._build_input_ui()
    def _on_exit(self):
        self._stop_anim()
        self.root.destroy()
    def _stop_anim(self):
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            self.anim = None
    def _clear_canvas(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
    # ══════════════════════════════════════════════════════════════
    #  LEER Y VALIDAR INPUTS
    # ══════════════════════════════════════════════════════════════
    def _on_simulate(self):
        body      = self.body_var.get()
        es_tierra = (body == 'Tierra')
        modo      = self.modo_var.get()
        if es_tierra:
            GM = G * M_TIERRA; escala = R_TIERRA
            unidad = 'R⊕'; r_factor = R_TIERRA
            R_central = R_TIERRA; nc = 'Tierra'
        else:
            GM = G * M_SOL; escala = UA
            unidad = 'UA'; r_factor = UA
            R_central = 0.0; nc = 'Sol'
        try:
            r1_u  = float(self.e_r1.get().strip())
            th1_d = float(self.e_th1.get().strip())
            r2_u  = float(self.e_r2.get().strip())
            th2_d = float(self.e_th2.get().strip())
        except ValueError:
            self._log("✗ ERROR: escribe solo números.")
            return
        r1  = r1_u  * r_factor
        th1 = np.radians(th1_d)
        r2  = r2_u  * r_factor
        th2 = np.radians(th2_d)
        if r1 <= 0 or r2 <= 0:
            self._log("✗ ERROR: r₁ y r₂ deben ser > 0.")
            return
        if abs(np.degrees(th1 - th2)) < 0.01:
            self._log("✗ ERROR: θ₁ y θ₂ son demasiado similares.")
            return
        N_orbits = 5.0
        if modo == 'espiral':
            try:
                N_orbits = float(self.e_nvueltas.get().strip())
                N_orbits = float(np.clip(N_orbits, 0.5, 100.0))
            except ValueError:
                self._log("✗ ERROR: N debe ser un número (0.5–100).")
                return
        params = dict(
            r1=r1, th1=th1, r2=r2, th2=th2,
            GM=GM, escala=escala, unidad=unidad,
            R_central=R_central, nombre_cuerpo=nc,
            es_tierra=es_tierra, modo=modo,
            N_orbits=N_orbits,
        )
        self.current_params = params
        self._run_with_params(params)
    # ══════════════════════════════════════════════════════════════
    #  DESPACHO SEGÚN MODO
    # ══════════════════════════════════════════════════════════════
    def _run_with_params(self, p: dict):
        """Despacha al método correcto según el modo de simulación."""
        if p.get('modo') == 'espiral':
            self._run_espiral(p)
        else:
            self._run_normal(p)
    # ══════════════════════════════════════════════════════════════
    #  MODO NORMAL (conservativo)
    # ══════════════════════════════════════════════════════════════
    def _run_normal(self, p: dict):
        r1 = p['r1']; th1 = p['th1']
        r2 = p['r2']; th2 = p['th2']
        GM = p['GM']; escala = p['escala']; unidad = p['unidad']
        R_central = p['R_central']; nc = p['nombre_cuerpo']
        es_tierra = p['es_tierra']
        if hasattr(self, 'btn_sim'):
            self.btn_sim.configure(state=tk.DISABLED, text="⏳  Calculando...")
        self.root.update()
        try:
            e, l, eps, pp, tipo = calcular_e_l(r1, th1, r2, th2, GM)
        except ValueError as ex:
            self._log(f"✗ {ex}")
            if hasattr(self, 'btn_sim'):
                self.btn_sim.configure(state=tk.NORMAL, text="▶  SIMULAR")
            return
        try:
            vr0, om0 = condiciones_iniciales(r1, th1, e, l, GM, eps)
        except Exception as ex:
            self._log(f"✗ Cond. iniciales: {ex}")
            if hasattr(self, 'btn_sim'):
                self.btn_sim.configure(state=tk.NORMAL, text="▶  SIMULAR")
            return
        t_max, dt, r_esc, r_imp = elegir_parametros_integracion(
            eps, pp, GM, R_central, es_tierra)
        self._log("─" * 30)
        self._log(f"TIPO: {tipo}")
        self._log(f"ε = {eps:.6f}")
        self._log(f"e = {e:.3e} J/kg")
        self._log(f"l = {l:.3e} m²/s")
        vt0  = l / r1
        vtot = np.sqrt(vr0**2 + vt0**2)
        self._log(f"vr₀= {vr0/1e3:.2f} km/s")
        self._log(f"vt₀= {vt0/1e3:.2f} km/s")
        self._log(f"|v₀|= {vtot/1e3:.2f} km/s")
        self._log(f"Δt = {dt:.2f} s  |  t_max = {t_max:.2e} s")
        self._log("Integrando con RK4...")
        self.root.update()
        try:
            t, r, th, vr, x, y, estado = integrar(
                r1, vr0, th1, GM, l, t_max, dt, r_esc, r_imp)
        except Exception as ex:
            self._log(f"✗ Integración: {ex}")
            if hasattr(self, 'btn_sim'):
                self.btn_sim.configure(state=tk.NORMAL, text="▶  SIMULAR")
            return
        self._log(f"Pasos: {len(t):,}  |  Estado: {estado}")
        if estado == 'impacto':
            v_imp = np.sqrt(vr[-1]**2 + (l / r[-1])**2)
            ang   = np.degrees(np.arctan2(abs(vr[-1]), abs(l / r[-1])))
            self._log(f"💥 IMPACTO detectado")
            self._log(f"  v_imp = {v_imp/1e3:.1f} km/s")
            self._log(f"  ángulo = {ang:.1f}° desde tangente")
            self._log(f"  tiempo = {t[-1]:.1f} s ({t[-1]/60:.1f} min)")
        elif eps < 1.0 - 1e-5:
            rev = (th[-1] - th[0]) / (2 * np.pi)
            self._log(f"Revoluciones: {rev:.3f}")
        if hasattr(self, 'btn_sim'):
            self.btn_sim.configure(state=tk.NORMAL, text="▶  SIMULAR")
        self._stop_anim()
        self._clear_canvas()
        self._sim_done = False
        self._build_animation(
            t, r, th, vr, x, y, l, GM, eps, tipo,
            escala, unidad, R_central, nc, estado, pp)
    # ══════════════════════════════════════════════════════════════
    #  MODO ESPIRAL DE COLAPSO (disipativo)
    # ══════════════════════════════════════════════════════════════
    def _run_espiral(self, p: dict):
        r1 = p['r1']; th1 = p['th1']
        r2 = p['r2']; th2 = p['th2']
        GM = p['GM']; escala = p['escala']; unidad = p['unidad']
        R_central = p['R_central']; nc = p['nombre_cuerpo']
        es_tierra = p['es_tierra']
        N_orbits  = p.get('N_orbits', 5.0)
        if hasattr(self, 'btn_sim'):
            self.btn_sim.configure(state=tk.DISABLED, text="⏳  Calculando...")
        self.root.update()
        # Órbita base (determina l₀, e₀, ε)
        try:
            e, l0, eps, pp, tipo = calcular_e_l(r1, th1, r2, th2, GM)
        except ValueError as ex:
            self._log(f"✗ {ex}")
            if hasattr(self, 'btn_sim'):
                self.btn_sim.configure(state=tk.NORMAL, text="▶  SIMULAR")
            return
        try:
            vr0, _ = condiciones_iniciales(r1, th1, e, l0, GM, eps)
        except Exception as ex:
            self._log(f"✗ Cond. iniciales: {ex}")
            if hasattr(self, 'btn_sim'):
                self.btn_sim.configure(state=tk.NORMAL, text="▶  SIMULAR")
            return
        # Radio de colapso: superficie del cuerpo central o 10 km
        if R_central > 0:
            r_colapso = R_central          # impacta con la Tierra
        elif not es_tierra:
            r_colapso = R_SOL              # impacta con el Sol
        else:
            r_colapso = 1.0e4              # 10 km fallback
        # Coeficiente de disipación y período inicial
        b, T0 = calcular_b_disipacion(r1, GM, N_orbits, r_colapso)
        # Parámetros de integración
        t_max = N_orbits * T0 * 2.5        # margen generoso
        dt    = T0 / 3000.0                # 3000 pasos por período inicial
        if es_tierra:
            dt = min(dt, 2.0)
        dt = np.clip(dt, 0.01, 8.0 * 3600.0)
        # Limitar pasos totales a 2 M para rendimiento
        N_est = int(t_max / dt)
        if N_est > 2_000_000:
            dt = t_max / 2_000_000
            self._log(f"  → dt ajustado a {dt:.2f} s (límite de pasos)")
        # Log
        self._log("─" * 30)
        self._log("🌀 ESPIRAL DE COLAPSO")
        self._log(f"Órbita base: {tipo}")
        self._log(f"ε = {eps:.6f}  |  N = {N_orbits:.1f}")
        self._log(f"b = {b:.3e} s⁻¹")
        self._log(f"T₀= {T0/DIA:.4f} días")
        vt0  = l0 / r1
        vtot = np.sqrt(vr0**2 + vt0**2)
        self._log(f"|v₀| = {vtot/1e3:.2f} km/s")
        self._log(f"Δt = {dt:.2f} s  |  t_max = {t_max:.2e} s")
        self._log("Integrando (RK4 + disipación)...")
        self.root.update()
        try:
            t, r, th, vr, ls, x, y, estado = integrar_espiral(
                r1, vr0, th1, l0, GM, b, t_max, dt, R_central, r_colapso)
        except Exception as ex:
            self._log(f"✗ Integración: {ex}")
            if hasattr(self, 'btn_sim'):
                self.btn_sim.configure(state=tk.NORMAL, text="▶  SIMULAR")
            return
        rev = (th[-1] - th[0]) / (2 * np.pi)
        self._log(f"Pasos: {len(t):,}  |  Estado: {estado}")
        self._log(f"Revoluciones totales: {rev:.2f}")
        if estado in ('colapso', 'impacto'):
            v_fin = np.sqrt(vr[-1]**2 + (ls[-1] / r[-1])**2)
            self._log(f"💥 {estado.upper()}")
            self._log(f"  r_fin = {r[-1]/escala:.4f} {unidad}")
            self._log(f"  v_fin = {v_fin/1e3:.1f} km/s")
            self._log(f"  t_fin = {t[-1]/DIA:.4f} días")
        if hasattr(self, 'btn_sim'):
            self.btn_sim.configure(state=tk.NORMAL, text="▶  SIMULAR")
        self._stop_anim()
        self._clear_canvas()
        self._sim_done = False
        self._build_animation_espiral(
            t, r, th, vr, ls, x, y, l0, GM, eps, tipo,
            escala, unidad, R_central, nc, estado, N_orbits, b, T0)
    # ══════════════════════════════════════════════════════════════
    #  ANIMACIÓN — Modo Normal
    # ══════════════════════════════════════════════════════════════
    def _build_animation(self, t, r, th, vr, x, y, l, GM, eps, tipo,
                         escala, unidad, R_central, nc, estado, p):
        N_orig = len(t)
        skip= max(1, N_orig // 600)
        xs= x[::skip] / escala;  ys  = y[::skip] / escala
        rs= r[::skip];            ts  = t[::skip] / DIA
        N   = len(xs)
        stela= max(20, N // 12)
        E_t= energia_especifica(r, vr, l, GM)
        e0= E_t[0]
        dE= (E_t - e0) / abs(e0) * 100.0 if abs(e0) > 1e-30 else E_t * 0
        th_dot = np.gradient(th, t)
        dL= (r**2 * th_dot - l) / abs(l) * 100.0
        td= t / DIA
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(9.8, 7.2), facecolor='#060a07')
        gs = GridSpec(3, 2, self.fig,
                      hspace=0.60, wspace=0.38,
                      left=0.08, right=0.97,
                      top=0.92, bottom=0.09)
        ax_t= self.fig.add_subplot(gs[:2, 0])
        ax_r= self.fig.add_subplot(gs[0, 1])
        ax_th= self.fig.add_subplot(gs[1, 1])
        ax_E= self.fig.add_subplot(gs[2, 0])
        ax_L= self.fig.add_subplot(gs[2, 1])
        for ax in [ax_t, ax_r, ax_th, ax_E, ax_L]:
            ax.set_facecolor('#070c08')
            ax.tick_params(colors='#446644', labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor('#1a3a1a')
            ax.grid(True, alpha=0.12, lw=0.5, color='#1a3a1a')
        extra = '  💥 IMPACTO' if estado == 'impacto' else ''
        self.fig.suptitle(
            f"ε = {eps:.5f}  →  {tipo}{extra}",
            fontsize=11, color=C['green'],
            family='monospace', fontweight='bold', y=0.97)
        # Gráficas estáticas
        ax_r.plot(td, r / escala, color=C['r_plot'], lw=1.1)
        if R_central > 0:
            ax_r.axhline(R_central / escala, color='deepskyblue',
                         ls='--', lw=0.9, alpha=0.7, label=f'R_{nc}')
            ax_r.legend(fontsize=6, facecolor='#080c08',
                        labelcolor='white', edgecolor='#1a3a1a')
        ax_r.set_xlabel('t [días]', color='#446644', fontsize=7)
        ax_r.set_ylabel(f'r [{unidad}]', color='#446644', fontsize=7)
        ax_r.set_title('r(t)', color='#66aa66', fontsize=9, family='monospace')
        ax_th.plot(td, np.degrees(th), color=C['th_plot'], lw=1.1)
        ax_th.set_xlabel('t [días]', color='#446644', fontsize=7)
        ax_th.set_ylabel('θ [°]', color='#446644', fontsize=7)
        ax_th.set_title('θ(t)', color='#66aa66', fontsize=9, family='monospace')
        ax_E.plot(td, dE, color=C['e_plot'], lw=0.9)
        ax_E.axhline(0, color='#ff4444', ls='--', lw=0.7, alpha=0.6)
        ax_E.set_xlabel('t [días]', color='#446644', fontsize=7)
        ax_E.set_ylabel('ΔE/E₀ [%]', color='#446644', fontsize=7)
        ax_E.set_title('Conserv. Energía', color='#66aa66',
                       fontsize=9, family='monospace')
        ax_L.plot(td, dL, color=C['l_plot'], lw=0.9)
        ax_L.axhline(0, color='#ff4444', ls='--', lw=0.7, alpha=0.6)
        ax_L.set_xlabel('t [días]', color='#446644', fontsize=7)
        ax_L.set_ylabel('Δl/l [%]', color='#446644', fontsize=7)
        ax_L.set_title('Conserv. Mom. Angular', color='#66aa66',
                       fontsize=9, family='monospace')
        # Trayectoria
        marg = max(np.max(np.abs(xs)), np.max(np.abs(ys))) * 1.30
        ax_t.set_xlim(-marg, marg); ax_t.set_ylim(-marg, marg)
        ax_t.set_aspect('equal')
        ax_t.set_xlabel(f'x [{unidad}]', color='#446644', fontsize=8)
        ax_t.set_ylabel(f'y [{unidad}]', color='#446644', fontsize=8)
        ax_t.set_title('Trayectoria Orbital',
                       color='#66aa66', fontsize=9, family='monospace')
        ax_t.plot(xs, ys, '-', color=C['orbit'], lw=0.4, alpha=0.12, zorder=1)
        if R_central > 0:
            Rc = R_central / escala
            ax_t.add_patch(mpatches.Circle(
                (0,0), Rc, color='deepskyblue', alpha=0.80, zorder=5, label=nc))
            ax_t.add_patch(mpatches.Circle(
                (0,0), Rc, fill=False, edgecolor='white', lw=1.2, zorder=6))
            ax_t.plot(0, 0, '+', color='white', ms=5, zorder=7)
        else:
            ax_t.plot(0, 0, 'o', color='#ffdd00', ms=14, zorder=5,
                      mec='orange', mew=1.5, label=nc)
        ax_t.plot(xs[0], ys[0], '^', color='lime', ms=9,
                  zorder=7, label='Inicio (t=0)')
        estela,    = ax_t.plot([], [], '-', color=C['orbit'],
                               lw=2.8, alpha=0.92, zorder=3)
        cuerpo_m,  = ax_t.plot([], [], 'o', color='white', ms=9, zorder=8,
                               mec=C['orbit'], mew=1.8, label='m (partícula)')
        radio,     = ax_t.plot([], [], '-', color='#2a4a2a',
                               lw=0.8, alpha=0.75, zorder=2)
        marca_imp, = ax_t.plot([], [], 'X', color='red', ms=15,
                               zorder=9, visible=False)
        info = ax_t.text(
            0.03, 0.97, '', transform=ax_t.transAxes,
            color=C['green'], fontsize=8, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='#030805', alpha=0.92, edgecolor='#1a3a1a'))
        vline_r= ax_r.axvline(0,  color=C['green'], lw=0.8, alpha=0.5)
        vline_th= ax_th.axvline(0, color=C['green'], lw=0.8, alpha=0.5)
        vline_E= ax_E.axvline(0,  color=C['green'], lw=0.8, alpha=0.5)
        vline_L= ax_L.axvline(0,  color=C['green'], lw=0.8, alpha=0.5)
        ax_t.legend(fontsize=7, facecolor='#070c08', labelcolor='white',
                    edgecolor='#1a3a1a', loc='lower right')
        imp_frame = N - 1 if estado == 'impacto' else None
        def init():
            estela.set_data([], [])
            cuerpo_m.set_data([], [])
            radio.set_data([], [])
            marca_imp.set_visible(False)
            info.set_text('')
            return (estela, cuerpo_m, radio, marca_imp, info,
                    vline_r, vline_th, vline_E, vline_L)
        def update(frame):
            i0 = max(0, frame - stela)
            estela.set_data(xs[i0:frame+1], ys[i0:frame+1])
            cuerpo_m.set_data([xs[frame]], [ys[frame]])
            radio.set_data([0, xs[frame]], [0, ys[frame]])
            t_actual = ts[frame]
            for vl in [vline_r, vline_th, vline_E, vline_L]:
                vl.set_xdata([t_actual, t_actual])
            v_r_i= abs(vr[min(frame * skip, len(vr)-1)])
            v_t_i=abs(l / r[min(frame * skip, len(r)-1)])
            v_tot=np.sqrt(v_r_i**2 + v_t_i**2)
            if imp_frame is not None and frame >= imp_frame:
                marca_imp.set_data([xs[-1]], [ys[-1]])
                marca_imp.set_visible(True)
                txt = (f't = {t_actual:.4f} días\n'
                       f'r = {rs[frame]/escala:.4f} {unidad}\n'
                       f'v = {v_tot/1e3:.2f} km/s\n'
                       f'💥 IMPACTO')
            else:
                txt = (f't = {t_actual:.4f} días\n'
                       f'r = {rs[frame]/escala:.4f} {unidad}\n'
                       f'v = {v_tot/1e3:.2f} km/s')
            info.set_text(txt)
            if frame == N - 1 and not self._sim_done:
                self._sim_done = True
                self.root.after(400, self._show_post_buttons)
            return (estela, cuerpo_m, radio, marca_imp, info,
                    vline_r, vline_th, vline_E, vline_L)
        self.anim = animation.FuncAnimation(
            self.fig, update, frames=N, init_func=init,
            interval=20, blit=True, repeat=False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    # ══════════════════════════════════════════════════════════════
    #  ANIMACIÓN — Modo Espiral de Colapso
    # ══════════════════════════════════════════════════════════════
    def _build_animation_espiral(self, t, r, th, vr, ls, x, y, l0, GM,
                                 eps, tipo, escala, unidad, R_central, nc,
                                 estado, N_orbits, b, T0):
        """
        Visualización para la espiral de colapso.
        La trayectoria se dibuja como un LineCollection con gradiente de
        color (plasma_r): morado = inicio, amarillo = colapso.
        Las cuatro gráficas muestran r(t), E(t), l(t) y θ(t).
        """
        N_orig = len(t)
        skip   = max(1, N_orig // 800)
        xs  = x[::skip] / escala;  ys  = y[::skip] / escala
        rs  = r[::skip];            ts  = t[::skip] / DIA
        lss = ls[::skip]
        N   = len(xs)
        stela = max(15, N // 15)
        td = t / DIA
        # Energía específica con l variable
        E_t  = 0.5 * vr**2 + 0.5 * (ls / r)**2 - GM / r
        e0   = E_t[0]
        dE   = ((E_t - e0) / abs(e0) * 100.0
                if abs(e0) > 1e-30 else np.zeros_like(E_t))
        l_norm = ls / ls[0]   # l(t)/l₀: decae exponencialmente
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(9.8, 7.2), facecolor='#060a07')
        gs = GridSpec(3, 2, self.fig,
                      hspace=0.62, wspace=0.42,
                      left=0.09, right=0.96,
                      top=0.92, bottom=0.09)
        ax_t  = self.fig.add_subplot(gs[:2, 0])   # trayectoria espiral
        ax_r  = self.fig.add_subplot(gs[0, 1])    # r(t)
        ax_E  = self.fig.add_subplot(gs[1, 1])    # disipación energía
        ax_l  = self.fig.add_subplot(gs[2, 0])    # decaimiento l(t)
        ax_th = self.fig.add_subplot(gs[2, 1])    # θ(t)
        for ax in [ax_t, ax_r, ax_E, ax_l, ax_th]:
            ax.set_facecolor('#080806')
            ax.tick_params(colors='#665544', labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor('#2a1a0a')
            ax.grid(True, alpha=0.12, lw=0.5, color='#2a1a0a')
        self.fig.suptitle(
            f"🌀 ESPIRAL DE COLAPSO  |  ε₀={eps:.4f}  |  N≈{N_orbits:.0f} vueltas"
            f"  |  b={b:.2e} s⁻¹",
            fontsize=10, color=C['orange'],
            family='monospace', fontweight='bold', y=0.97)
        # ── Trayectoria con gradiente de color (LineCollection) ────
        xf = x / escala;  yf = y / escala
        # Submuestreo para LC (máx ~5000 segmentos para rendimiento)
        sk_lc   = max(1, len(xf) // 5000)
        xf_lc   = xf[::sk_lc];  yf_lc = yf[::sk_lc]
        pts     = np.array([xf_lc, yf_lc]).T.reshape(-1, 1, 2)
        segs    = np.concatenate([pts[:-1], pts[1:]], axis=1)
        c_vals  = np.linspace(0, 1, len(segs))
        lc      = LineCollection(segs, cmap='plasma_r', array=c_vals,
                                 lw=1.6, alpha=0.60, zorder=2)
        ax_t.add_collection(lc)
        marg = max(np.max(np.abs(xf)), np.max(np.abs(yf))) * 1.20
        ax_t.set_xlim(-marg, marg);  ax_t.set_ylim(-marg, marg)
        ax_t.set_aspect('equal')
        ax_t.set_xlabel(f'x [{unidad}]', color='#665544', fontsize=8)
        ax_t.set_ylabel(f'y [{unidad}]', color='#665544', fontsize=8)
        ax_t.set_title('Espiral de Colapso  (color = tiempo)',
                       color=C['orange'], fontsize=9, family='monospace')
        # Colorbar temporal
        cbar = self.fig.colorbar(lc, ax=ax_t, pad=0.02, fraction=0.034)
        cbar.set_label('t / t_final', color='#665544', fontsize=6)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['inicio', 'mitad', 'fin'], fontsize=6)
        cbar.ax.yaxis.set_tick_params(color='#665544')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#665544')
        # Cuerpo central
        if R_central > 0:
            Rc = R_central / escala
            ax_t.add_patch(mpatches.Circle(
                (0,0), Rc, color='deepskyblue', alpha=0.80, zorder=5))
            ax_t.add_patch(mpatches.Circle(
                (0,0), Rc, fill=False, edgecolor='white', lw=1.2, zorder=6))
            ax_t.plot(0, 0, '+', color='white', ms=5, zorder=7)
            ax_t.text(0, Rc*1.15, nc, color='deepskyblue',
                      fontsize=7, ha='center', family='monospace')
        else:
            ax_t.plot(0, 0, 'o', color='#ffdd00', ms=14,
                      zorder=5, mec='orange', mew=1.5, label=nc)
        ax_t.plot(xs[0], ys[0], '^', color='lime', ms=8,
                  zorder=7, label='Inicio')
        # ── Gráficas estáticas ────────────────────────────────────
        # r(t): decaimiento radial
        ax_r.plot(td, r / escala, color='#cc88ff', lw=1.1)
        if R_central > 0:
            ax_r.axhline(R_central / escala, color='deepskyblue',
                         ls='--', lw=0.9, alpha=0.7, label=f'R_{nc}')
            ax_r.legend(fontsize=6, facecolor='#080806',
                        labelcolor='white', edgecolor='#2a1a0a')
        ax_r.set_xlabel('t [días]', color='#665544', fontsize=7)
        ax_r.set_ylabel(f'r [{unidad}]', color='#665544', fontsize=7)
        ax_r.set_title('r(t) — decaimiento radial',
                       color='#66aa66', fontsize=9, family='monospace')
        # E(t): disipación de energía
        ax_E.plot(td, dE, color=C['e_plot'], lw=0.9)
        ax_E.axhline(0, color='#ff4444', ls='--', lw=0.7, alpha=0.6)
        ax_E.set_xlabel('t [días]', color='#665544', fontsize=7)
        ax_E.set_ylabel('ΔE/E₀ [%]', color='#665544', fontsize=7)
        ax_E.set_title('Disipación de Energía',
                       color=C['orange'], fontsize=9, family='monospace')
        # l(t): decaimiento exponencial del momento angular
        ax_l.plot(td, l_norm, color=C['l_plot'], lw=1.1, label='simulado')
        t_th  = np.linspace(0, td[-1], 500)
        l_th  = np.exp(-b * t_th * DIA)
        ax_l.plot(t_th, l_th, '--', color='white', lw=0.8,
                  alpha=0.55, label='exp(−b·t)')
        ax_l.legend(fontsize=6, facecolor='#080806',
                    labelcolor='white', edgecolor='#2a1a0a')
        ax_l.set_xlabel('t [días]', color='#665544', fontsize=7)
        ax_l.set_ylabel('l(t) / l₀', color='#665544', fontsize=7)
        ax_l.set_title('Decaimiento Mom. Angular',
                       color=C['orange'], fontsize=9, family='monospace')
        # θ(t): ángulo acumulado
        ax_th.plot(td, np.degrees(th), color=C['th_plot'], lw=1.1)
        ax_th.set_xlabel('t [días]', color='#665544', fontsize=7)
        ax_th.set_ylabel('θ [°]', color='#665544', fontsize=7)
        ax_th.set_title('θ(t)',
                        color='#66aa66', fontsize=9, family='monospace')
        # ── Elementos animados ─────────────────────────────────────
        estela,= ax_t.plot([], [], '-', color='white',
                              lw=2.8, alpha=0.90, zorder=3)
        cuerpo_m, = ax_t.plot([], [], 'o', color='white', ms=10,
                              zorder=8, mec=C['orange'], mew=2.2,
                              label='m (partícula)')
        marca_fin, = ax_t.plot([], [], 'X', color='#ff2244',
                               ms=18, zorder=9, visible=False)
        vline_r= ax_r.axvline(0,  color=C['orange'], lw=0.8, alpha=0.6)
        vline_E= ax_E.axvline(0,  color=C['orange'], lw=0.8, alpha=0.6)
        vline_l= ax_l.axvline(0,  color=C['orange'], lw=0.8, alpha=0.6)
        vline_th= ax_th.axvline(0, color=C['orange'], lw=0.8, alpha=0.6)
        info = ax_t.text(
            0.03, 0.97, '', transform=ax_t.transAxes,
            color=C['orange'], fontsize=8, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='#050302', alpha=0.92,
                      edgecolor='#3a2a0a'))
        ax_t.legend(fontsize=7, facecolor='#080806',
                    labelcolor='white', edgecolor='#2a1a0a',
                    loc='lower right')
        def init():
            estela.set_data([], [])
            cuerpo_m.set_data([], [])
            marca_fin.set_visible(False)
            info.set_text('')
            return (estela, cuerpo_m, marca_fin, info,
                    vline_r, vline_E, vline_l, vline_th)
        def update(frame):
            # Estela blanca corta sobre la trayectoria degradada
            i0 = max(0, frame - stela)
            estela.set_data(xs[i0:frame+1], ys[i0:frame+1])
            cuerpo_m.set_data([xs[frame]], [ys[frame]])
            t_actual = ts[frame]
            for vl in [vline_r, vline_E, vline_l, vline_th]:
                vl.set_xdata([t_actual, t_actual])
            idx= min(frame * skip, N_orig - 1)
            l_inst= ls[idx]
            vr_inst= abs(vr[idx])
            vt_inst= abs(l_inst / r[idx])
            v_tot= np.sqrt(vr_inst**2 + vt_inst**2)
            frac_l= l_inst / l0
            txt = (f't = {t_actual:.4f} días\n'
                   f'r = {rs[frame]/escala:.4f} {unidad}\n'
                   f'|v| = {v_tot/1e3:.2f} km/s\n'
                   f'l/l₀ = {frac_l:.4f}')
            if frame == N - 1:
                if estado in ('colapso', 'impacto'):
                    txt += f'\n💥 {estado.upper()}'
                marca_fin.set_data([xs[-1]], [ys[-1]])
                marca_fin.set_visible(True)
            info.set_text(txt)
            if frame == N - 1 and not self._sim_done:
                self._sim_done = True
                self.root.after(400, self._show_post_buttons)
            return (estela, cuerpo_m, marca_fin, info,
                    vline_r, vline_E, vline_l, vline_th)
        self.anim = animation.FuncAnimation(
            self.fig, update, frames=N, init_func=init,
            interval=18, blit=True, repeat=False)
        self.canvas= FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
# ═══════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    root = tk.Tk()
    try:
        root.iconbitmap('orbit.ico')
    except Exception:
        pass
    app = OrbitalSimulator(root)
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    ww, wh = 1420, 820
    x = (sw - ww) // 2
    y = (sh - wh) // 2
    root.geometry(f"{ww}x{wh}+{x}+{y}")
    root.mainloop()