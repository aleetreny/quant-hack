# Ronda 1: cómo quedar primero en solitario frente al grupo de clones EV-DP

Supuestos: rewards Exp(10), T=2000, L=15. El ranking es el mejor score sobre las runs. En cada run
todos ven la misma secuencia y el "grupo" son clones de la EV-DP, que empatan entre sí al céntimo.
El mejor actual es B0 = 2800 y quedan K ∈ {10, 20, 40, 60} runs. **Ganar** significa que nuestro mejor
score supera estrictamente al mejor final del grupo, max(B0, max de sus K runs).
Escrito el 2026-11-04.

## Recomendación

**No uses TARGET. Usa "shadow-lock":** sigue en vivo el score `g` que la EV-DP del grupo lleva en
esta run (vemos las mismas recompensas, así que basta con simular su regla) y juega así:

```
gap = umbral EV de esta ronda (la misma expresión exacta que usa la EV-DP: reward >= gap)
kappa = 1.0 si (your_score - g) > 0, si no KAPPA_BEHIND = 1.1
accept = reward >= kappa * gap                # si estamos bloqueados, el servidor lo ignora
# después, actualiza la sombra con la regla del grupo:
if t > g_lock_until and reward >= gap: g += reward; g_lock_until = t + L
```

Mientras vamos empatados o por detrás (D = s − g ≤ 0), un umbral un 10% más exigente nos separa de la
trayectoria del grupo. En cuanto vamos por delante volvemos a la EV exacta. La siguiente vez que los
dos estemos libres y aceptemos la misma recompensa, los bloqueos se realinean y desde ahí jugamos
igual que el grupo: la ventaja D > 0 queda congelada hasta el final de la run. La media apenas cae
(2685 frente a 2690), y en cada run P(y > x) = 0.83 frente a P(y < x) = 0.17.

| K restantes | shadow-lock κ_b=1.1 | EV×1.025 (sin sombra) | utility-DP | mejor target fijo | mejor target dinámico/por cuantiles |
|---|---|---|---|---|---|
| 10 | **0.731** / 0.269 | 0.495 / 0.494 | 0.500 / 0.493 | 0.330 (B=2920) / 0.670 | 0.342 / 0.658 |
| 20 | **0.783** / 0.217 | 0.533 / 0.453 | 0.539 / 0.453 | 0.370 (B=2970) / 0.630 | 0.383 / 0.617 |
| 40 | **0.788** / 0.212 | 0.541 / 0.445 | 0.550 / 0.446 | 0.398 (B=3000) / 0.602 | 0.412 / 0.588 |
| 60 | **0.788** / 0.212 | 0.546 / 0.438 | no ejecutado* | 0.409 (B=3010) / 0.591 | 0.429 / 0.571 |

Cada celda es P(ganar) / P(acabar por debajo del grupo); lo que falta hasta 1 es empate. Son 40 000
concursos bootstrap sobre pools de 200k-400k runs independientes. La utility-DP son 10 000 concursos
pareados, simulados run a run. El error estándar es ≤ 0.0025 en la columna bootstrap y 0.005 en la
utility.
\* No ejecuté la utility-DP con K=60 (unos 10 minutos más de cálculo). Con K = 10, 20 y 40 va a la
par que EV×1.025 (+0.005 a +0.009), muy por debajo del shadow-lock, así que no cambia la conclusión.

Si se prefiere seguir con la target-DP (por ejemplo, si no se confía en la sombra): TARGET ≈ 3000
con K = 40-60 y TARGET ≈ 2950 con K = 10-20. Mejor aún, la regla por cuantiles de la sección 1c.
Aun así, cualquier target pierde contra el grupo más del 57% de las veces.

### Riesgos del shadow-lock (léelos antes de integrarlo)
1. **La rama "por delante" tiene que ser exactamente la regla EV del grupo.** Ese es el mecanismo de
   congelación. Si usamos κ_a = 0.99 o 1.01 en esa rama, P(ganar) cae a 0.62-0.69.
2. **La sombra supone que el grupo usa nuestra misma EV.** Si su umbral difiere un 0.1%, P(ganar)
   baja a 0.69-0.76. Si difiere un 1%, baja a 0.58-0.63 (umbral del grupo ×0.99) o a 0.52-0.56
   (×1.01), y aun así está por encima de cualquier target. Tabla en `_local/r1/shadow.txt`.
   Comprobación barata en vivo: si `other_scores` trae los scores de los demás equipos de la run,
   varios deberían coincidir al céntimo con `g`. Si no coinciden, la sombra está mal calibrada.
3. Comparar `your_score - g > 0` en float: D = 0 (sincronizados) cuenta como "detrás o empatado",
   y eso es justo lo que queremos, porque hay que desviarse.

## 1. Target fijo frente a B dinámico (pool, 40k concursos, B0 = 2800)

Ver `r1_fixed.py`, `r1_rules.py` y `r1_rules2.py`. La salida completa está en `_local/r1/{fixed,rules,rules2}.txt`.

**1a. Target fijo.** El óptimo está cerca de la media del mejor final del grupo:

| K | media del mejor final del grupo | B óptimo | P(ganar) | P(ganar) con 2950 / 3000 / 3050 |
|---|---|---|---|---|
| 10 | 2896 | 2920 | 0.330 | 0.322 / 0.297 / 0.276 |
| 20 | 2937 | 2970 | 0.370 | 0.372 / 0.355 / 0.319 |
| 40 | 2977 | 3000 | 0.398 | 0.361 / 0.400 / 0.362 |
| 60 | 2999 | 3010 | 0.409 | 0.333 / 0.406 / 0.394 |

**1b. B dinámico.** B = X + δ, con X el mejor del grupo hasta ahora, o max(X, Y) + δ si incluye
nuestro mejor. Es peor que el fijo con K ≥ 20: un B bajo al principio gasta runs en superar valores
que el grupo superará después. Los mejores resultados son max(X, Y) + δ con δ = 60-90: 0.335 / 0.364
/ 0.371 / 0.381 para K = 10 / 20 / 40 / 60.

**1c. Regla por cuantiles según las runs que quedan (la mejor de las de target).** Con m runs
restantes (incluida esta):

`TARGET = max( X, Q(0.7^(1/m)) , max(X, Y) + 25 )`

Aquí Q es el cuantil del score EV-DP. Q(0.7^(1/m)) es el cuantil 70 de max(X, M_m), donde M_m es el
mejor de m runs EV. Resultados: P(ganar) = 0.338 / 0.379 / 0.411 / 0.427. Es robusta si m está mal
estimado por un factor 2 (pierde ≤ 1.4 puntos).

| m restantes | 1 | 2 | 3 | 5 | 8 | 10 | 15 | 20 | 30 | 40 | 50 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q(0.7^(1/m)) | 2756 | 2817 | 2848 | 2883 | 2914 | 2928 | 2952 | 2968 | 2990 | 3004 | 3015 | 3024 | 3038 | 3049 |

El score EV-DP tiene media 2689.9, sd 129.2 y asimetría 0.086. Su cola es algo más pesada que una
normal (q99 2999 frente a 2990; q99.9 3109 frente a 3089). Cuantiles empíricos: q90 2856, q95 2905,
q97 2937, q99 2999, q99.5 3032, q99.9 3109.

**Mezclar runs EV puras para asegurar el empate** (1 de cada e runs; la última run siempre EV).
Cambia victorias por empates: con e = 4 y K = 40 queda 0.327 / 0.202 / 0.471
(ganar / empatar / perder). No compensa salvo que el empate valga casi tanto como ganar.

## 2. Políticas mejores que la target-DP pura

**Utility-DP** (`r1_lib.value_tables`, `r1_contest.py`). Es la misma DP O(D) por paso, pero con la
condición terminal V_0(s) = U(s) = 1[max(Y, s) > X] · G(max(Y, s))^(c·m) en lugar de 1[s ≥ B]. G es
la CDF empírica del score EV-DP y la tabla se reconstruye cada run. Está validada: el valor de la DP
es 0.0874 y el Monte Carlo da 0.0875. Gana claramente a la target-DP (+17 a +20 puntos en concursos
pareados) con c = 1 (c = 0.5 es ~1 punto peor). El motivo: la política óptima es **casi la EV-DP**.
Una target-DP se aleja mucho de la trayectoria del grupo: con B = 3000, corr(y, x) = 0.83 y en las
runs del 1% superior del grupo P(y > x) = 0.18. Una política casi EV (κ = 1.025) mantiene
corr = 0.95 y P(y > x | x en el 1% superior) = 0.39, y además suma las runs casi récord. Aun
así, U trata el score del grupo en esta misma run como independiente del nuestro, así que no explota
lo que sigue.

**EV × κ** (`r1_kappa.py`). La versión de una línea de esa idea: aceptar si r ≥ κ·gap. El óptimo es
κ ≈ 1.025-1.03 y empata con la utility-DP. Con κ < 1 sale peor: ser más exigente rinde más en las runs
afortunadas. P(ganar) / P(perder) con κ = 1.025: 0.495 / 0.494, 0.533 / 0.453, 0.541 / 0.445,
0.546 / 0.438.

**Shadow-lock** (`r1_shadow.py`). Es la recomendación de arriba. La gran ganancia sale de usar
información que la DP 1D no ve: el score del grupo en esta misma run. Barrido de κ_b (con κ_a = 1,
grupo exacto, K = 40): 1.025 → 0.723, 1.05 → 0.774, 1.07 → 0.787, **1.1 → 0.788**, 1.13 → 0.783,
1.16 → 0.771, 1.2 → 0.747, 1.5 → 0.528, 0.97 → 0.601.

Posible mejora (no hecha): una DP 2D sobre (n, D, bloqueo propio, bloqueo del grupo) con terminal
1[D > 0] elegiría el momento óptimo de desviarse y podría pasar de 0.79.

## 3. Coste de los empates / acabar por debajo

- **EV-DP pura**: P(ganar) = 0 y P(debajo) = 0. Empatamos siempre en mejor score y también en
  "latest score" (misma secuencia), así que la posición la decide el nombre ASCII dentro del grupo.
  Solo quedamos primeros si nuestro nombre va antes que el de todos los clones.
- **Target-DP**: casi nunca empata (< 0.1%). Acaba por debajo del grupo, es decir detrás de todos los
  clones, el 57-67% de las veces (tablas de arriba).
- **EV × 1.025**: empate 1-1.5% (runs en las que nunca se separa de la EV), debajo 44-49%.
- **Shadow-lock 1.1**: empate ≈ 0 y debajo 21-27%. Es la opción con menor P(debajo) entre las que
  pueden ganar.

Si el premio es solo para el primero, shadow-lock domina. Si el premio va por posición, el
compromiso es este: con P = 0.21 acabamos detrás de los clones, frente a 0 con la EV pura (que en
cambio nunca gana sola).

## 4. Tiempos (local, M-series)

- `target_tables(B)` de 3000 a 3700: 0.24-0.33 s en frío y hasta 0.48 s con la máquina cargada.
- `value_tables` (utility, D = 3700): 0.24-0.36 s.
- Las dos se construyen una vez por partida (t ≤ 2). Repartido entre 300 llamadas son ~1-1.5 ms por
  llamada, por debajo del límite de 15 ms.
- **Shadow-lock**: sin tabla, ~0.2 µs por llamada en Python puro.

## Notas de método

- Las políticas de una run dependen solo de esa run, así que los concursos se montan remuestreando
  runs de un pool (`r1_pool.py`, 400k runs × rejilla de B cada 10 entre 2800 y 3400). La utility-DP
  depende del estado (X, Y, m), así que se simula run a run con recompensas compartidas
  (`r1_contest.py`, 10k concursos pareados con los target como control; sus cifras coinciden con las
  del pool).
- Un error que hay que evitar: si se guardan los scores en float32 y se comparan con los del grupo en
  float64, los empates exactos (runs en las que la política nunca se separa de la EV) se cuentan como
  victorias o derrotas. Los scores de `r1_kappa.py` y `r1_shadow.py` se guardan en float64.
