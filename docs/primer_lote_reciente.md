# Primer lote reciente: captura y predicción local

## Estado

Los notebooks 04–07 ya fueron ejecutados por Diego. El 05 verificó una
inferencia histórica; el 06 comprobó el guardado sin reemplazo; el 07 importó
esa función desde `src/ingestion/capture_store.py`.

El notebook 08 ya produjo un lote real completo de cinco licitaciones para
la consulta del 14 de octubre de 2026. Se recuperaron dos detalles y se
reutilizaron tres. El CSV y registro quedaron guardados, las probabilidades
son válidas y los 35 archivos originales observados conservaron sus huellas.
Las cinco clases predichas fueron 0; esto verifica el recorrido, no la calidad
del modelo. El resultado inicial se conserva íntegro.
Las huellas de código de ese primer registro describen la versión usada
entonces. La mejora de consultas cambia únicamente el cliente y coordinador;
los siguientes lotes registrarán esas nuevas versiones de código.

La mejora de cadencia pasó dieciocho pruebas automáticas con respuestas,
relojes y vectores simulados; su operación real sigue pendiente. No se
realizaron nuevas consultas API ni inferencia para verificar esta mejora.

## Qué entrega

Una consulta de fecha y hasta cinco detalles por defecto (máximo diez),
TextoLicitacion preparado con las funciones existentes, embeddings E5 y el
clasificador registrado como candidato. Genera un CSV de predicciones y un
registro que permite relacionarlo con las capturas y el modelo utilizado.

El listado se ordena por código y se toman los primeros códigos únicos, sin
filtro temático. Es una prueba del recorrido; no representa el universo ni
garantiza encontrar oportunidades relevantes. La API puede devolver menos
casos o ninguno. Consultar una fecha no garantiza que todas las licitaciones
sigan abiertas: se conserva el estado recibido y su fecha de captura.

## Recorrido y archivos

```text
API por fecha → listado.json → hasta cinco consultas de detalle
                                  ↓
                   details/<codigo>.json + captura.json
                                  ↓
          normalizar → TextoLicitacion → E5 → candidato MLflow
                                  ↓
                  predicciones.csv + registro_ejecucion.json
```

Capturas nuevas:
`data/raw/batches/<lote_id>/`. Se conservan listado, detalles y registro con
selección, fecha UTC, huellas, cantidad de consultas y fallos.

Resultados nuevos:
`data/processed/batches/<lote_id>/<intento_id>/`. El CSV conserva código, título,
organismo, región, fechas, estado capturado, texto, cuatro probabilidades,
afinidad predicha y esperada, margen, fuente, huellas e identidad del modelo.
El registro agrega versiones de bibliotecas y huellas del código reutilizado.
La tabla visible en el notebook muestra sólo las columnas para revisar.

Los datos quedan en disco. No son copias temporales como en los notebooks
anteriores. Las carpetas `data/raw` y `data/processed` ya están excluidas de
Git; este trabajo no publica información.

## Cómo ejecutar

1. Abrir el proyecto como carpeta raíz en VS Code y elegir su kernel `.venv`.
2. Abrir `notebooks/08_lote_reciente_captura_y_prediccion.ipynb`.
3. Ejecutar la configuración: propone ayer según la fecha del PC y límite 5.
   Revisar la fecha mostrada; puede fijarse explícitamente con formato ddmmaaaa.
4. Ejecutar captura: aquí sí se consulta la API con el cliente y ticket locales.
   El programa no muestra el ticket ni guarda URLs con credenciales.
   Cada petición espera al menos tres segundos desde el fin de la anterior.
   Si aparece HTTP 429, permite un solo reintento: espera Retry-After cuando
   está disponible, o treinta segundos en su ausencia. Un segundo 429 o una
   espera indicada mayor de sesenta segundos corta las solicitudes restantes
   del lote y conserva los detalles ya guardados. Son valores iniciales del
   proyecto, no una cuota oficial ni garantía de evitar 429.
5. Ejecutar predicción: usa los JSON guardados, no vuelve a consultar la API.
6. Revisar la tabla, guardar el notebook y compartir el resumen de salida.

No hay instalaciones ni entrenamiento. El encoder debe estar disponible en la
caché local. El catálogo MLflow se copia mediante SQLite abierto en modo de
sólo lectura; los motores de esa copia se cierran y su carpeta temporal se
retira. Se resuelve el alias y se carga la versión fija del comprobante (1);
si cambian versión, run, firma o parámetros aprendidos, la ejecución se detiene.
Se verifica que modelo, catálogo, comprobante, artefactos, perfil y código
observados mantengan sus huellas antes de exportar.

## Si se interrumpe

- **Día vacío:** queda registrado; no se fabrican detalles ni predicciones.
  Elegir otra fecha constituye una captura nueva.
- **Fallo de API o detalle inesperado:** queda un lote fallido o incompleto,
  con las capturas que sí llegaron. No se predice una selección parcial.
  Para un lote incompleto, `recuperar_lote(carpeta)` crea un lote derivado,
  conserva los detalles correctos y consulta una sola vez cada código fallido.
  No repite la consulta del listado. Si vuelve a fallar, conserva el registro
  nuevo con el motivo y el estado HTTP cuando el cliente lo proporciona.
  No guarda URLs ni mensajes de error arbitrarios. Cada consulta puede tener
  un reintento 429; no hay reintentos ilimitados. El registro cuenta peticiones
  HTTP efectivas, incluidos reintentos, y distingue códigos omitidos por espera.
  Si el lote conserva una fecha de espera futura, recuperar_lote la respeta
  antes de efectuar otra consulta. El control es por lote; aún no coordina
  múltiples procesos o capturas nuevas ejecutadas en paralelo.
  Un lote fallido antes de tener selección requiere revisar la causa y luego
  repetir captura; no puede recuperarse con esta función.
- **Fallo durante la predicción:** conserva las capturas y el registro del
  intento. Tras resolver el problema, se puede repetir sólo predicción sobre
  la misma carpeta. No es necesario repetir captura.
- **Predicción ya completada:** rechaza otra ejecución sobre ese lote para
  evitar un CSV duplicado. Se abre el resultado existente.
- **Kernel reiniciado:** recuperar la ruta del lote desde la salida guardada
  y asignarla a `CARPETA_LOTE = Path(r'...')`; ejecutar sólo predicción, o abrir
  el CSV ya terminado. No hace falta repetir la celda de captura.

### Recuperar desde el notebook que ya está abierto

Después de actualizar `recent_batch.py`, agregar una celda debajo de la captura
fallida. Recargar el módulo permite usar la corrección sin perder el kernel:

```python
import importlib
from pathlib import Path
import src.ingestion.recent_batch as modulo_lote

modulo_lote = importlib.reload(modulo_lote)
CARPETA_LOTE = modulo_lote.recuperar_lote(Path(r'RUTA_DEL_LOTE_INCOMPLETO'))
predecir_lote = modulo_lote.predecir_lote
```

Cuando termine en estado `completo`, continuar con la celda 3 del notebook 08.
El lote original permanece intacto. El derivado mantiene la procedencia y las
fechas originales de los detalles reutilizados; los recuperados tienen su
propia fecha. Si vuelve a fallar, revisar el motivo impreso y conservar la ruta
nueva. No ejecutar otra vez la captura inicial para intentar arreglarlo.

## Qué significa el resultado

Cada fila es una predicción pendiente de revisión humana; conserva también
la clase 0. El valor esperado ordena las cuatro probabilidades como en el
piloto anterior. No representa probabilidad de adjudicación ni una etiqueta
humana confirmada. Una licitación con varios ítems recibe una sola predicción.

Después del primer lote conviene revisar algunas filas de mayor afinidad y
algunas de clase 0, contrastándolas con su texto y criterio. No se modifican
etiquetas históricas a partir de predicciones.

### Primera revisión confirmada

El usuario confirmó afinidad 3 para 1003780-1-LE26, Nueva Alameda, el
16 de octubre de 2026. El motivo es el componente de análisis, segmentación
e indicadores de seguimiento, según el criterio 0.2.0. La revisión es asistida
y confirmada por el usuario sobre el texto disponible. A petición del usuario
no se completó la lectura de las bases. No equivale a confirmar conveniencia
comercial ni capacidad de ejecutar todo el contrato.

Se registra separadamente en
`data/processed/reviews/20261016T030234991480Z_5b2b8bbd/revision_001.csv`.
AfinidadPredicha conserva 0 y AfinidadHumana registra 3, con huellas del CSV,
texto, captura y criterio. Las otras cuatro filas siguen sin revisión humana.
Esta confirmación no se incorpora a train, validación, test ni al registro
MLflow, y no genera métricas nuevas de calidad.

## Próximos pasos, después de comprobar esta ejecución

El historial acumulativo local y la vista de seguimiento están implementados;
ver [Historial acumulativo](historial_acumulativo.md). Conservan capturas,
predicciones y revisiones por separado, con comprobación de huellas e
idempotencia. La revisión de Nueva Alameda permanece separada del entrenamiento.

Siguen pendientes la coordinación de un nuevo lote con la actualización del
historial, la ejecución semanal, BigQuery y la evaluación mensual para decidir
si corresponde reentrenar. No hay scheduler ni reentreno automático.
