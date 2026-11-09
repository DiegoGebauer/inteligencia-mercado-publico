# Captura incremental: varios segmentos por corrida

El ejecutor nuevo `src.run_incremental_batch` coordina el ejecutor incremental
existente. La fuente de Looker sigue siendo `vista_actual` en BigQuery.

## Instalar manualmente

Copiar al proyecto estos archivos del paquete, respetando sus carpetas:

1. `src/run_incremental.py`: reemplaza el actual. Conserva su CLI de un segmento.
2. `src/run_incremental_batch.py`: archivo nuevo.
3. `tests/test_incremental_batch.py`: archivo nuevo.
4. `docs/ejecucion_incremental_por_corridas.md`: esta guía.

`referencia/run_incremental_anterior.py` es una copia del motor anterior para
comparar; no instalarla como otro módulo del proyecto. Si has editado el motor
desde que se preparó el paquete, compara los cambios antes de reemplazarlo.

El motor actualizado acepta un contador de API compartido y un bloqueo ya
adquirido por el coordinador. Sus comandos anteriores siguen funcionando. El
coordinador verifica la versión de esa interfaz antes de ejecutar.

## Comprobaciones que ejecuta el usuario

Primero, todas las pruebas:

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q tests
```

Las pruebas nuevas no consultan la API, no cargan modelos y no escriben en
BigQuery. Los escenarios de ejecución usan respuestas y operaciones simuladas.

Después, el plan con una fecha nueva y dos que ya tienen listado guardado:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.run_incremental_batch --desde 2026-11-05 --hasta 2026-11-07 --limite 10 --max-detalles 30 --max-segmentos 3 --orden recientes
```

Sin `--ejecutar` sólo lee y valida los archivos existentes. No escribe el diario
ni el reporte, no carga el modelo ni contacta API/BigQuery. No puede anticipar
los códigos o tamaños de una fecha cuyo listado todavía no se ha consultado.

Con las entradas revisadas el 8 de noviembre, el orden inicial es 7, 6 y 5 de
noviembre. El día 7 está pendiente de consultar. El 6 tiene cinco códigos con
predicción local y el 5 tiene veinte.

## Ejecutar una corrida, después de revisar pruebas y plan

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.run_incremental_batch --desde 2026-11-05 --hasta 2026-11-07 --limite 10 --max-detalles 30 --max-segmentos 3 --orden recientes --ejecutar --publicar-bigquery
```

Se detiene al alcanzar el primer límite aplicable o ante un fallo. Cada
segmento captura una fecha, predice y confirma su publicación antes de iniciar
otro. No se deduce éxito sólo de tener el CSV local.

| Parámetro | Significado |
|---|---|
| `--limite 10` | Máximo de detalles nuevos por segmento; de 1 a 10. |
| `--max-detalles 30` | Hasta 30 códigos que se intentan consultar en esta corrida. Un fallo también consume presupuesto. |
| `--max-segmentos 3` | Hasta tres operaciones; una recuperación o un día vacío también ocupa una. |
| `--orden recientes` | Empieza por las fechas más recientes del rango. |
| `--orden antiguas` | Empieza por las más antiguas para una carga retrospectiva. |
| `--max-consultas` | Máximo de solicitudes HTTP, incluidos reintentos; por defecto `2 × (max-detalles + max-segmentos)`, 66 en el ejemplo. |
| `--max-segundos 600` | Deja de iniciar segmentos cuando se agota el tiempo. Permite terminar el que ya está activo. |
| `--max-bytes` | Límite de procesamiento de la consulta BigQuery del cargador, por publicación; no es un presupuesto monetario de toda la corrida. |

Antes de iniciar un segmento se reserva el peor caso de un reintento por cada
consulta lógica. Por eso puede terminar con solicitudes sin utilizar si el
segmento siguiente no cabe en esa reserva. La pausa y los reintentos por 429 se
comparten entre segmentos, sin reiniciar la cadencia en cada uno. Son controles
internos, no cuotas oficiales de Mercado Público.

Después de completar una fecha, la desplaza al final de la cola de esta corrida.
Así, con tres fechas y tres segmentos, intenta atender una vez cada fecha antes
de repetir alguna. Las fechas agotadas se saltan. Una operación pendiente de
una corrida anterior tiene prioridad sobre los datos nuevos.

La cola vuelve a su orden inicial al comenzar otra corrida. Si el rango es más
largo que la capacidad, todavía no se garantiza atender todas las fechas en cada
corrida. Para producción se deberán separar un rango corto de actualización
reciente y las cargas retrospectivas, con presupuestos explícitos para ambos.

## Revisión del resultado

Al terminar aparece un resumen con las operaciones completadas, los lotes, las
fechas, las solicitudes HTTP reales, los detalles consultados y el motivo de
parada. Se guarda también en:

```text
data/processed/incremental/ejecuciones/<identificador>.json
```

El reporte incluye las huellas del coordinador y del motor. Los resultados del
modelo siguen conservando la trazabilidad del capturador anterior. Un estado
`completo` en el reporte significa que la corrida terminó según sus límites,
no que todo el rango ni todo el mercado estén cubiertos.

Después de una ejecución confirmada, volver a lanzar el comando sin `--ejecutar`
permite revisar los pendientes. En Looker, ampliar el período y actualizar los
datos permite ver las nuevas capturas. El número de licitaciones únicas puede
crecer menos que el de detalles consultados si un código ya existía en otra fecha.

## Fallos y recuperación

- Ante un fallo no inicia otro segmento. Los anteriores ya confirmados se
  conservan.
- El diario de operación sigue en `data/processed/incremental/estado.json`.
- Capturas incompletas e inferencias pendientes conservan la recuperación del
  ejecutor anterior. Las copias de detalles correctos no consumen de nuevo el
  presupuesto de consultas.
- Una publicación sin confirmación exige revisar el job de BigQuery antes de
  cualquier reintento. Después de revisar su resultado, se puede usar el mismo
  comando real añadiendo `--reintentar-publicacion`. Esa corrida sólo reanuda la
  publicación pendiente y termina; no inicia capturas nuevas.
- Un cierre forzado puede dejar un bloqueo o un lote sin manifiesto final. El
  ejecutor se detiene para revisarlo. No borrar archivos mientras un proceso
  siga activo.
- El bloqueo abarca toda la corrida, pero sólo coordina estos ejecutores. No
  lanzar `run_pipeline` o el ejecutor individual en paralelo.

## Qué falta para cloud

Esta entrega no despliega servicios, no programa tareas y no reentrena. Sigue
usando archivos locales y el candidato MLflow actual. Falta portar el estado y
los artefactos a almacenamiento persistente, empaquetar el modelo/encoder y
configurar la ejecución y sus permisos. También falta la política de refresco
de listados y recaptura de licitaciones ya conocidas.

Los listados se conservan como instantáneas de una fecha consultada: no aseguran
que las oportunidades estén abiertas ahora ni que su publicación pertenezca a
ese día. No se cambió la selección alfabética ni se añadieron filtros comerciales
que descarten códigos antes de inferir.

## Validación al preparar la entrega

Se comprobó la sintaxis y el plan en modo lectura contra las entradas locales
existentes. No se ejecutaron pruebas del proyecto, capturas reales, inferencias
ni publicaciones cloud durante la preparación. Las pruebas y la primera corrida
real las ejecuta el usuario antes de guardar este avance en Git.
