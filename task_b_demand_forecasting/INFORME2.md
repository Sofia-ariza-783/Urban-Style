# Informe — Tarea B: Pronóstico de Demanda con LSTM

**Proyecto:** UrbanStyle — Retail de moda
**Tarea:** B — Demand Forecasting (Time-Series con LSTM pequeño / encoder-decoder)
**Autor(es):** [COMPLETAR]
**Fecha:** [COMPLETAR]

---

## 1. Contexto de negocio

UrbanStyle es un retailer de moda que necesita anticipar la demanda de sus productos para tomar mejores decisiones de inventario. Si compra de más, el capital queda inmovilizado y aumentan los descuentos de liquidación. Si compra de menos, se pierden ventas por falta de stock.

Las compras de inventario no se deciden semana a semana, sino con varias semanas de anticipación. Por eso el objetivo de esta tarea es **pronosticar las ventas de las próximas semanas por categoría de producto** con una red neuronal recurrente (LSTM), medir qué tan confiables son esas predicciones y **traducirlas a una explicación en lenguaje natural** que el equipo de compras pueda leer sin interpretar gráficas.

## 2. Objetivos

**Requeridos por el enunciado:**
1. Seleccionar entre 5 y 8 series (categorías) de ventas.
2. Transformar las transacciones en **series de tiempo semanales**.
3. Entrenar un **LSTM pequeño o un encoder-decoder**.
4. Pronosticar las ventas futuras.
5. Evaluar el modelo con **MAE** y **RMSE**.
6. Construir una interfaz interactiva en **Pygame**.

**Opcionales y extensiones implementadas:**

7. Comparar con un **baseline de media móvil**.
8. **Exportar las predicciones a CSV**.
9. **Regularización y early stopping**, para controlar el sobreajuste.
10. **Pronóstico de varias semanas** (1 a 6) con arquitectura encoder-decoder.
11. **Explicación automática del pronóstico** generada por un modelo de lenguaje (Claude).

## 3. Datos

**Fuente:** [Retail Sales Dataset — Kaggle](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset)

| Característica | Valor |
|---|---|
| Registros (transacciones) | [COMPLETAR — p. ej. 1.000] |
| Periodo | [COMPLETAR — p. ej. ene-2023 a ene-2024] |
| Categorías de producto | Beauty, Clothing, Electronics |
| Columnas usadas | `Date`, `Gender`, `Age`, `Product Category`, `Quantity`, `Total Amount` |

**Variable objetivo:** `Total Amount`, el ingreso total por transacción, sumado por semana.

### 3.1 Definición de las series

El enunciado pide entre 5 y 8 categorías, pero el dataset trae solo **3 categorías de producto**. Para cumplir el requisito sin inventar datos, cada categoría se dividió por **género del cliente**, lo que da **6 series**:

| # | Serie |
|---|---|
| 1 | Beauty - Female |
| 2 | Beauty - Male |
| 3 | Clothing - Female |
| 4 | Clothing - Male |
| 5 | Electronics - Female |
| 6 | Electronics - Male |

Esta segmentación también tiene sentido para el negocio, porque la oferta y la demanda en moda suelen gestionarse por línea (dama y caballero). El código permite cambiarla en `config.py` (`SEGMENTATION`): `category` da 3 series y `category_age`, por grupos de edad, da 9.

## 4. Preprocesamiento

1. **Conversión de fechas:** la columna `Date` se pasa a formato fecha.
2. **Agregación semanal:** se suma `Total Amount` por serie y por semana. Las semanas terminan en lunes (`W-MON`).
3. **Semanas sin ventas:** se rellenan con 0 para que la serie sea continua.
4. **División temporal en tres bloques**, siempre en orden cronológico:

```
|◄──────────── entrenamiento ────────────►|◄─ validación (6) ─►|◄── prueba (8) ──►|
```

| Bloque | Uso |
|---|---|
| Entrenamiento | Ajustar los pesos del modelo |
| Validación (6 semanas) | Decidir cuándo detener el entrenamiento (early stopping) |
| Prueba (8 semanas) | Medir el error final; el modelo nunca la ve antes |

5. **Escalado min-max a [0, 1]:** se ajusta **solo con los datos de entrenamiento**, para que no se filtre información del futuro.
6. **Ventanas deslizantes:** cada muestra usa `lookback = 4` semanas como entrada y las `horizon = 4` semanas siguientes como objetivo.

```
entrada: [ s1, s2, s3, s4 ]  →  objetivo: [ s5, s6, s7, s8 ]
entrada: [ s2, s3, s4, s5 ]  →  objetivo: [ s6, s7, s8, s9 ]
...
```

**Resultado del preprocesamiento:** [COMPLETAR — p. ej. "53 semanas por serie, 6 series"; se obtiene con `python data_prep.py`]

## 5. Modelo: encoder-decoder LSTM

### 5.1 Arquitectura

Se usó un **encoder-decoder** (también llamado *sequence-to-sequence*) implementado en PyTorch, que es una de las dos opciones que propone el enunciado. A diferencia de un LSTM simple, que predice una sola semana, este modelo **genera varias semanas futuras de una vez**.

```
 Últimas 4 semanas            Estado oculto            Próximas 4 semanas
[s1][s2][s3][s4] ──► ENCODER ──► (h, c) ──► DECODER ──► [ŷ1] → [ŷ2] → [ŷ3] → [ŷ4]
                      LSTM                     LSTM        ▲      │      │
                                                           └──────┴──────┘
                                              cada predicción alimenta el paso siguiente
```

1. **Encoder:** un LSTM lee las últimas `lookback` semanas y resume el patrón reciente en su estado oculto `(h, c)`.
2. **Decoder:** otro LSTM parte de ese estado y genera las semanas futuras **una a la vez**. La predicción de cada semana se usa como entrada para la siguiente.
3. **Teacher forcing (50%):** durante el entrenamiento, la mitad de las veces el decoder recibe el valor **real** de la semana anterior en lugar de su propia predicción. Esto acelera el aprendizaje y evita que los errores tempranos se acumulen.

| Componente | Configuración |
|---|---|
| Entrada | Secuencia de 4 semanas × 1 variable |
| Encoder | LSTM, 1 capa, 32 neuronas |
| Decoder | LSTM, 1 capa, 32 neuronas |
| Salida | Dropout (0.2) + capa lineal (32 → 1), aplicada en cada paso |
| Horizonte | 4 semanas por defecto (configurable de 1 a 6) |
| Pérdida | Error cuadrático medio (MSE) sobre todo el horizonte |
| Optimizador | Adam, tasa de aprendizaje 0.01, weight decay 1e-4 |
| Semilla aleatoria | 42, para que los resultados sean reproducibles |

Se entrena **un modelo independiente por cada serie**. Se eligió un modelo **pequeño** (32 neuronas) a propósito, porque con unas 50 semanas de historia uno grande se aprendería el ruido de memoria.

### 5.2 Regularización y early stopping

Con tan pocos datos, el mayor riesgo es el **sobreajuste**: que el modelo memorice el ruido del entrenamiento y falle con datos nuevos. Se aplicaron tres técnicas:

| Técnica | Qué hace | Valor |
|---|---|---|
| **Dropout** | Apaga neuronas al azar durante el entrenamiento, para que el modelo no dependa de unas pocas | 0.2 |
| **Weight decay (L2)** | Penaliza los pesos grandes, lo que produce un modelo más simple | 1e-4 |
| **Early stopping** | Detiene el entrenamiento cuando el error en validación deja de mejorar | Paciencia de 30 épocas |

Así funciona el early stopping:
1. En cada época se mide el error en el bloque de validación.
2. Se guarda una copia del modelo cada vez que ese error mejora.
3. Si pasan **30 épocas seguidas sin mejora**, el entrenamiento se detiene (el máximo es 500 épocas).
4. Se recupera **la copia de la mejor época**, no la última.
5. Las **primeras 20 épocas no se tienen en cuenta**. En las pruebas, con tan pocas semanas de validación, el error de la época 1 salía engañosamente bajo y el modelo quedaba sin entrenar.

## 6. Métricas de evaluación

Ambas métricas están en la misma unidad que las ventas (dólares), así que se interpretan directamente. Siendo $y_t$ la venta real, $\hat{y}_t$ la predicción y $n$ el número de semanas de prueba:

$$\text{MAE} = \frac{1}{n}\sum_{t=1}^{n} |y_t - \hat{y}_t|$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$

- **MAE:** cuánto se equivoca el modelo, en promedio, cada semana.
- **RMSE:** castiga más los errores grandes. Si el RMSE es mucho mayor que el MAE, hay semanas con errores muy fuertes.

### 6.1 Cómo se evalúa un pronóstico de varias semanas

Se usan dos formas complementarias:

1. **Pronóstico por bloques (métrica principal):** se simula el uso real. Desde el inicio del periodo de prueba se pronostican 4 semanas, luego se avanza 4 semanas y se vuelve a pronosticar. Sobre esas predicciones se calculan el MAE y el RMSE generales.
2. **Error según cuántas semanas adelante:** desde cada semana de prueba se pronostica el horizonte completo y se mide el MAE a 1, 2, 3 y 4 semanas. Lo esperado es que **el error crezca cuanto más lejos se pronostica**.

### 6.2 Baseline: media móvil

Para saber si el modelo aporta valor, se comparó con un método simple: **pronosticar que cada una de las próximas semanas venderá el promedio de las últimas 4**. Se evalúa con la misma lógica de bloques. Un modelo de machine learning solo se justifica si supera, o al menos iguala, a una regla tan sencilla.

## 7. Resultados

> Se obtienen ejecutando `python train.py`. Los valores quedan en `outputs/metrics_summary.csv` y `outputs/metrics_by_horizon.csv`.

### 7.1 Métricas por serie (periodo de prueba: últimas 8 semanas)

| Serie | MAE LSTM | RMSE LSTM | MAE Media móvil | RMSE Media móvil | Mejor época | Épocas ejecutadas | Mejor modelo |
|---|---|---|---|---|---|---|---|
| Beauty - Female | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Beauty - Male | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Clothing - Female | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Clothing - Male | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Electronics - Female | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Electronics - Male | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| **Promedio** | **[ ]** | **[ ]** | **[ ]** | **[ ]** | | | |

### 7.2 Error según cuántas semanas adelante se pronostica

*Valores de la tabla "MAE promedio según cuántas semanas adelante" que imprime `train.py`.*

| Semanas adelante | MAE LSTM | MAE Media móvil |
|---|---|---|
| 1 | [ ] | [ ] |
| 2 | [ ] | [ ] |
| 3 | [ ] | [ ] |
| 4 | [ ] | [ ] |

### 7.3 Pronóstico de las próximas 4 semanas

*Valores de las filas `forecast` de `outputs/predictions_all.csv`.*

| Serie | Semana 1 | Semana 2 | Semana 3 | Semana 4 | Total |
|---|---|---|---|---|---|
| Beauty - Female | [ ] | [ ] | [ ] | [ ] | [ ] |
| Beauty - Male | [ ] | [ ] | [ ] | [ ] | [ ] |
| Clothing - Female | [ ] | [ ] | [ ] | [ ] | [ ] |
| Clothing - Male | [ ] | [ ] | [ ] | [ ] | [ ] |
| Electronics - Female | [ ] | [ ] | [ ] | [ ] | [ ] |
| Electronics - Male | [ ] | [ ] | [ ] | [ ] | [ ] |

### 7.4 Visualización

![Pronóstico vs real — app Pygame](report/img/app_forecast.png)

*Figura 1. La línea gris son las ventas reales, la naranja el pronóstico del encoder-decoder y la verde la media móvil. Las zonas sombreadas son validación (morado), prueba (naranja) y pronóstico futuro (verde). En prueba, los puntos grandes marcan el inicio de cada bloque de pronóstico.*

![Curvas de pérdida](report/img/app_loss.png)

*Figura 2. Pérdida de entrenamiento (azul) y validación (morado) por época. La línea vertical marca la mejor época, la que se conservó con early stopping.*

### 7.5 Experimentos con hiperparámetros (opcional)

| Configuración | Comando | MAE promedio LSTM | RMSE promedio LSTM |
|---|---|---|---|
| Base (horizonte 4, lookback 4) | `python train.py` | [ ] | [ ] |
| Horizonte corto | `python train.py --horizon 1` | [ ] | [ ] |
| Horizonte largo | `python train.py --horizon 6` | [ ] | [ ] |
| Ventana más larga | `python train.py --lookback 6` | [ ] | [ ] |
| Más paciencia | `python train.py --patience 60` | [ ] | [ ] |
| Menos historia | `python train.py --history-weeks 35` | [ ] | [ ] |

## 8. Explicación automática del pronóstico con IA

### 8.1 Motivación

Una gráfica de pronóstico requiere interpretación: ¿las ventas suben o bajan?, ¿cuánto?, ¿qué tan confiable es? Para que el resultado sea útil al equipo de compras, al exportar cada pronóstico se genera también un **documento en lenguaje natural** que explica el comportamiento esperado del mercado y **cómo se llegó a esas conclusiones**.

### 8.2 Funcionamiento

```
Resultado del modelo ──► 1. Python calcula los datos clave ──► 2. Claude redacta ──► explicacion_<serie>.md
                            (tendencia, % de cambio,              (o redactor por
                             semanas pico, errores)                reglas, sin internet)
```

1. **Cálculo de los datos (`build_facts`):** todo lo numérico se calcula en Python, no lo calcula el modelo de lenguaje.
   - Promedio histórico y de las últimas 4 semanas.
   - Pronóstico semana por semana y su cambio porcentual frente al promedio reciente.
   - Tendencia del horizonte, a partir de la pendiente de una recta ajustada al pronóstico: al alza, a la baja o estable, con un umbral de ±5% semanal.
   - Semanas con el valor más alto y más bajo.
   - MAE como porcentaje de la venta promedio, comparación con la media móvil y error según semanas adelante.
   - Datos del entrenamiento: mejor época y si se activó el early stopping.
2. **Redacción con Claude:** esos datos se envían en JSON a la API de Anthropic (modelo `claude-opus-5-5`) con instrucciones explícitas:
   - usar solo las cifras recibidas;
   - presentar las posibles causas como hipótesis;
   - reportar con honestidad si la media móvil fue mejor.
3. **Estructura del documento**, siempre con las mismas cinco secciones:
   - Resumen
   - Comportamiento esperado en las próximas semanas
   - Cómo se llegó a estas conclusiones
   - Qué tan confiable es este pronóstico
   - Recomendación para inventario
4. **Trazabilidad:** al final de cada documento se adjuntan los datos JSON que se usaron, para que cualquier afirmación del texto se pueda verificar.
5. **Respaldo sin conexión:** si no hay API key o falla la conexión, un **redactor por reglas** genera el documento con la misma estructura. El encabezado indica qué motor lo generó.

### 8.3 Decisiones de diseño

| Decisión | Motivo |
|---|---|
| Separar el cálculo (Python) de la redacción (LLM) | Los modelos de lenguaje pueden equivocarse con la aritmética o inventar cifras. Así, los números siempre son correctos y el LLM solo redacta. |
| Adjuntar los datos al documento | Transparencia: el lector puede comprobar cada afirmación. |
| Redactor de respaldo por reglas | La aplicación funciona sin internet ni costo; por ejemplo, durante la presentación. |
| Generación en segundo plano | La llamada a la API tarda unos segundos; la interfaz de Pygame no se congela. |

### 8.4 Ejemplo de explicación generada

> [COMPLETAR — pegar aquí un fragmento de un `outputs/explicacion_<serie>.md` real, p. ej. las secciones "Resumen" y "Cómo se llegó a estas conclusiones".]

## 9. Interfaz Pygame

La aplicación (`python app.py`) permite explorar los pronósticos de forma interactiva:

| Control | Función |
|---|---|
| Botones de serie | Elegir la categoría que se pronostica (reentrena automáticamente) |
| Slider **Máx. épocas** | Tope de épocas (50–1000); el early stopping suele cortar antes |
| Slider **Lookback** | Cuántas semanas anteriores lee el encoder (2–8) |
| Slider **Horizonte** | Cuántas semanas pronostica el decoder (1–6) |
| Slider **Semanas de historia** | Cuánta historia se usa para entrenar |
| Botón **Entrenar** | Reentrenar con la configuración elegida |
| Botón **Media móvil** | Mostrar u ocultar el baseline |
| Botón **Exportar CSV + explicación** | Guardar las predicciones y generar su explicación escrita |

La pantalla muestra cuatro paneles:
1. **Gráfica principal:** ventas reales, pronóstico del modelo y media móvil, con las zonas de validación, prueba y pronóstico.
2. **Métricas:** MAE y RMSE del LSTM y de la media móvil, el MAE según semanas adelante y el resultado del early stopping.
3. **Próximas semanas:** tabla con el pronóstico de cada semana y el total del horizonte.
4. **Curvas de pérdida:** entrenamiento y validación, con la mejor época marcada.

![Interfaz Pygame](report/img/app_ui.png)

*Figura 3. Interfaz completa de la aplicación.*

## 10. Discusión

> Ajustar según los resultados reales. Abajo van guías para la redacción.

- **¿Superó el LSTM a la media móvil?** [COMPLETAR — p. ej. "El encoder-decoder obtuvo menor MAE en X de 6 series..."]
- **Efecto del horizonte:** [COMPLETAR — tabla 7.2: ¿cuánto crece el error de 1 a 4 semanas? ¿El LSTM degrada más lento o más rápido que la media móvil?]
- **Efecto del early stopping:** [COMPLETAR — ¿en qué época se detuvo cada serie? Las curvas de la Figura 2 muestran si el error de validación empezaba a subir (sobreajuste) cuando se detuvo]
- **Series más predecibles y menos predecibles:** [COMPLETAR — cuáles tienen menor y mayor error, y por qué podría ser: más o menos volumen, más o menos volatilidad]
- **MAE frente a RMSE:** [COMPLETAR — si el RMSE es bastante mayor que el MAE, hay semanas con picos que el modelo no anticipa, por ejemplo compras grandes de Electronics]
- **Calidad de las explicaciones generadas:** [COMPLETAR — ¿el texto de Claude fue fiel a los datos? ¿Agregó valor frente al redactor por reglas?]

**Interpretación esperada:** con cerca de un año de datos (unas 53 semanas) y pocas transacciones por semana, las series son **cortas y ruidosas**. En ese escenario es común que un método simple como la media móvil compita con el LSTM o le gane, sobre todo a horizontes largos. Es un hallazgo válido: muestra que **la complejidad del modelo debe ir de la mano con la cantidad de datos disponibles**, y justifica las técnicas de regularización aplicadas.

## 11. Limitaciones

1. **Poca historia:** cerca de 53 semanas por serie no alcanzan para capturar estacionalidad anual, que requeriría al menos 2 o 3 años.
2. **Bajo volumen por serie:** al dividir por género, cada serie tiene pocas transacciones por semana, lo que aumenta el ruido.
3. **Validación pequeña:** 6 semanas dan pocas ventanas de validación, así que la mejor época puede variar con la semilla aleatoria.
4. **Una sola variable de entrada:** el modelo solo ve las ventas pasadas, sin promociones, precios, festivos ni clima.
5. **Primera semana parcial:** los datos empiezan a mitad de semana, así que la primera semana tiene menos ventas de las reales.
6. **Explicación con IA:** requiere internet y una API key. Su calidad depende del modelo de lenguaje, aunque las cifras siempre vienen de Python.

## 12. Posibles modificaciones a futuro

- Entrenar un **modelo global** con las 6 series a la vez, para aprovechar más datos.
- Agregar **variables externas**: festivos, temporadas como Black Friday o Navidad, precio promedio.
- Mostrar **intervalos de confianza** (por ejemplo, con un ensamble de semillas) y pasarlos a la explicación como rango esperado.
- Usar **validación cruzada temporal** (walk-forward) para medir el error de forma más robusta.
- Pronosticar **unidades** (`Quantity`) además de ingresos. Basta con cambiar `TARGET_COLUMN` en `config.py`.

## 13. Reproducibilidad

```bash
cd task_b_forecasting
python -m venv .venv
.venv\Scripts\activate            # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# Colocar el CSV de Kaggle en data/retail_sales_dataset.csv
$env:ANTHROPIC_API_KEY="sk-ant-..."   # opcional: explicación con Claude
python train.py --explain         # métricas, CSV y explicaciones en outputs/
python app.py                     # interfaz Pygame
```

Todas las ejecuciones usan semilla fija (`SEED = 42`), así que las métricas se repiten. La configuración completa está en `config.py`. El texto generado por Claude puede variar ligeramente entre ejecuciones; las cifras que cita no.

---

## Anexo A — Cumplimiento de la rúbrica (Tarea B, 20%)

| Criterio | Estado | Evidencia |
|---|---|---|
| Seleccionar 5–8 categorías | ✅ | 6 series (categoría × género), sección 3.1 |
| Preprocesar como serie semanal | ✅ | Sección 4, `data_prep.py` |
| Entrenar un LSTM pequeño o encoder-decoder | ✅ | Encoder-decoder LSTM, sección 5, `model.py` |
| Pronosticar ventas futuras | ✅ | Próximas 1–6 semanas, sección 7.3 |
| Evaluar con MAE y RMSE | ✅ | Secciones 6 y 7 |
| Pygame: gráfica y sliders | ✅ | Sección 9, `app.py` |
| *Opcional:* baseline de media móvil | ✅ | Secciones 6.2 y 7 |
| *Opcional:* exportar predicciones a CSV | ✅ | `outputs/predictions_*.csv` |
| *Extensión:* regularización y early stopping | ✅ | Sección 5.2, `forecasting.py` |
| *Extensión:* explicación automática con IA | ✅ | Sección 8, `narrator.py` |

## Anexo B — Estructura del código

| Archivo | Descripción |
|---|---|
| `config.py` | Rutas e hiperparámetros |
| `data_prep.py` | Carga del CSV y construcción de las series semanales |
| `model.py` | Encoder-decoder LSTM |
| `forecasting.py` | Entrenamiento con early stopping, métricas, baseline y exportación |
| `narrator.py` | Explicación del pronóstico (Claude o reglas) |
| `train.py` | Ejecución por terminal de todas las series |
| `app.py` | Interfaz Pygame |
| `make_sample_data.py` | Datos sintéticos, solo para pruebas |
