# Historial acumulativo local

## Qué quedó implementado

`src/reporting/build_batch_history.py` reúne los lotes, capturas, predicciones
y revisiones que ya están guardados. Sólo usa la biblioteca estándar de
Python. Importarlo no ejecuta el trabajo; no consulta la API, carga modelos,
genera nuevas predicciones ni modifica las fuentes o etiquetas históricas.

El módulo pasó 15 pruebas con archivos sintéticos. También se ejecutó sobre
los archivos reales disponibles y se repitió sobre las mismas entradas:
devolvió la misma carpeta y conservó exactamente los mismos bytes de salida.

Resultado verificado con los archivos disponibles al 18 de octubre de 2026:

| Elemento | Cantidad |
|---|---:|
| Lotes registrados | 4 |
| Capturas distintas | 8 |
| Referencias desde lotes a capturas | 14 |
| Predicciones guardadas | 5 |
| Revisiones humanas confirmadas | 1 |
| Licitaciones en la vista actual | 7 |

Se reconocieron seis referencias reutilizadas sin duplicar las capturas.
Dos licitaciones tienen captura válida pero no predicción, porque proceden
de un lote incompleto; la vista las muestra como pendientes. No se calcula
una predicción automáticamente para completar esos huecos.

Nueva Alameda conserva afinidad predicha 0 y afinidad humana 3. Su revisión
es asistida y confirmada por el usuario; no se incorpora al entrenamiento.

## Cómo ejecutarlo

Desde la raíz del proyecto, con su entorno:

```powershell
.\.venv\Scripts\python.exe -B -m src.reporting.build_batch_history
```

El comando imprime la carpeta verificada y sus conteos. También puede
llamarse desde una sesión Python que ya tenga disponible la raíz del proyecto:

```python
from src.reporting.build_batch_history import build_history
carpeta_historial = build_history()
```

Para consultar la vista con pandas:

```python
import pandas as pd
vista = pd.read_csv(carpeta_historial / 'vista_actual.csv', dtype={'CodigoExterno': 'string'})
vista[['CodigoExterno', 'EstadoCapturado', 'AfinidadPredicha', 'AfinidadHumana', 'EstadoRevision']]
```

Esta consulta reutiliza tu aprendizaje de pandas para inspeccionar un resultado
del proyecto. No hace falta crear otro notebook para ejecutar el módulo.

## Archivos de salida

Cada versión queda en `data/processed/history/<huella>/`:

| Archivo | Para qué sirve |
|---|---|
| `vista_actual.csv` | Una fila por código, basada en su captura más reciente. |
| `capturas.csv` | Cada respuesta válida distinta, con su fecha y huella. |
| `lote_captura.csv` | Qué lotes usan cada captura, incluidas las reutilizaciones. |
| `predicciones.csv` | Cada predicción guardada, con modelo, captura y probabilidades. |
| `revisiones.csv` | Revisiones vinculadas a una predicción concreta. |
| `lotes.csv` | Registro de lotes completos, incompletos, vacíos o fallidos. |
| `resumen.json` | Conteos y huellas de todas las entradas y salidas. |

La vista muestra el estado recibido en la captura; no consulta el estado actual
de Mercado Público. Las fechas de publicación y cierre se conservan como
vinieron. Los datos de contacto de la respuesta no se exportan en estas tablas.

## Reglas para conservar el historial

Una captura se identifica por código, fecha de captura normalizada a UTC y
huella del JSON. Copiar esa respuesta durante una recuperación no crea otra
captura. Consultar de nuevo en otra fecha sí conserva un evento nuevo, aunque
el contenido sea idéntico.

Una predicción se identifica por lote, intento y código. Repetir la construcción
del índice no la duplica; otra ejecución real de inferencia conserva su propia
identidad. Las revisiones se identifican por predicción y RevisionId. Una copia
idéntica no se duplica; valores distintos para esa misma identidad se rechazan.

La vista elige la captura más reciente y, dentro de ella, la predicción más
reciente. Una revisión sólo se muestra junto a la predicción exacta que
referencia. No se traslada automáticamente a otra captura, texto o predicción.
Para varias revisiones de la misma predicción se usa FechaRevisionLocal y
RevisionId; deben seguir el orden revision_001, revision_002, etc. Las anteriores
permanecen en revisiones.csv.

## Repetición e integridad

La huella de la carpeta depende de las entradas y de la versión del constructor.
Si todo coincide, el módulo comprueba las salidas existentes y devuelve esa
carpeta. Si se agregan lotes o revisiones, crea una versión nueva con el historial
completo y conserva la anterior. No concatena archivos a ciegas.

Comprueba las huellas de capturas, listados, CSV de predicciones y referencias
de las revisiones, así como códigos, clases y probabilidades. Si una entrada
cambia mientras se lee, o una salida guardada fue alterada, se detiene.
Una carpeta incompleta por interrupción se conserva y se informa; no se borra
ni reemplaza automáticamente.

La primera versión usa instantáneas CSV verificables. La coordinación de
escrituras entre procesos, el programador semanal, BigQuery y el reentreno
siguen fuera de esta implementación. El siguiente paso es conectar una nueva
ejecución de lote con la actualización del historial y definir el flujo de
revisión humana de los casos pendientes.
