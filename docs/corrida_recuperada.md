# Primera corrida desde el estado recuperado de Google Cloud

## Qué prueba

Usa la carpeta preparada por bootstrap_cloud: modelo descargado y datos
recuperados, junto con el código que ya se validó. Lanza el ejecutor incremental
en un proceso independiente con esa carpeta como directorio de trabajo.

Sin --ejecutar muestra el plan. Con --ejecutar SÍ consulta Mercado Público,
calcula predicciones portables y publica en BigQuery. La prueba admite hasta
cinco detalles nuevos y un único segmento, conservando el límite de llamadas y
esperas del cliente existente. No ofrece cobertura total de los días elegidos.

Al finalizar comprueba que el diario corresponde a una finalización nueva y una
publicación BigQuery confirmada. Entonces crea un checkpoint verificable del
workspace. Todavía no sube ese paquete a Cloud Storage ni cambia la referencia
del proyecto principal.

## Preparación

Copiar los archivos de esta entrega al proyecto PRINCIPAL, desde donde se han
ejecutado los comandos anteriores. No copiar archivos manualmente al workspace:
el lanzador ejecuta los módulos que ya contiene esa copia validada.

Requiere un bootstrap.json con estado preparado, el modelo descargado y el diario
incremental completo. Verifica las huellas del código/configuración recuperados,
el modelo y la compatibilidad del constructor de historial antes de empezar.

## Pruebas y plan

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q tests/test_run_recovered.py
& '.\.venv\Scripts\python.exe' -B -m src.operations.run_recovered --desde 2026-11-05 --hasta 2026-11-07 --limite 5
```

El plan no lee .env, no crea registros ni carga modelos y no contacta servicios.
Invoca el plan existente de run_incremental_batch desde el workspace para mostrar
qué fecha y códigos siguen pendientes. Verifica que el modo sea portable_local.

## Ejecutar una sola corrida

Si el plan termina correctamente:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.operations.run_recovered --desde 2026-11-05 --hasta 2026-11-07 --limite 5 --ejecutar
```

Hereda ADC del Python actual para BigQuery. Para Mercado Público utiliza
MERCADO_PUBLICO_TICKET del entorno; si no está definido, lee únicamente ese valor
del .env del proyecto principal y lo pasa al entorno del proceso hijo. No copia
.env al workspace, no pasa el ticket por argumentos y no lo escribe en reportes.
No cambia las variables de la terminal o del proceso padre.

Activa MP_MODEL_BUNDLE únicamente en el hijo, apuntando al modelo descargado.
El comando usa un segmento, max-detalles igual al límite (máximo cinco), un
presupuesto HTTP de 2*(limite+1) y 300 segundos para INICIAR segmentos; no es un
timeout que mate un segmento activo. Mantiene los límites del cargador BigQuery.

El workspace acumula los datos nuevos y BigQuery se actualiza. Los datos de la
carpeta principal no se sincronizan de vuelta automáticamente. Para continuar,
usar este workspace o el próximo checkpoint; no alternar con el historial local
anterior del proyecto principal ni ejecutar otros escritores en paralelo.

## Resultado esperado

Después del resumen de la corrida incremental se muestra:

- ruta del checkpoint local dentro del workspace;
- identificador, tamaño y SHA256;
- conteos actualizados e identificador del job BigQuery;
- checkpoint_publicado_gcs: false.

El reporte del lanzador se guarda en <arranque>/corridas/<identificador>.json,
fuera del árbol de datos del workspace. Conserva la identidad del checkpoint de
entrada y la salida, el job confirmado y la huella del lanzador.

El guardado local aún no actualiza el bucket. El paso siguiente será subir ese
checkpoint y registrar su nueva generación; el respaldo anterior sigue disponible.

## Fallos y alcance

Conserva capturas, predicciones, diario y reporte si falla la corrida. Si BigQuery
no queda confirmado o no cambia el diario, no declara éxito ni crea un checkpoint
de finalización. Una corrida fallida/interrumpida bloquea otro lanzamiento para
evitar repetir consultas sin revisar lo ocurrido. No borra bloqueos o registros
automáticamente: revisar el diario y cualquier job BigQuery antes de reintentar.

El bloqueo del lanzador coordina sólo esta carpeta local. No es una exclusión
distribuida entre instancias cloud. Esta prueba sigue ejecutándose en tu PC;
quedan pendientes persistencia cloud automática, coordinación, contenedor Linux,
Cloud Run, programación y monitoreo.
