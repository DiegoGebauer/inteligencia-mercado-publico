# Checkpoint del estado incremental — primera versión

Objetivo: conservar fuera del PC los datos y el avance de una corrida terminada.
Es un respaldo recuperable; todavía no es sincronización automática ni una
solución de coordinación entre trabajadores cloud.

## Qué conserva

- Capturas de `data/raw/batches/`, incluidos los listados y lotes incompletos
  históricos que tengan un manifiesto válido.
- Predicciones y registros de `data/processed/batches/`.
- Revisiones y los criterios que referencian.
- Diario incremental y registros de corridas.
- El historial que referencia el diario y su comprobante local de publicación.
- Perfil de empresa y referencia versionada del modelo en GCS.

Cada archivo conserva sus bytes originales y su ruta relativa. El manifiesto
registra tamaños, SHA256, conteos y huellas de cuatro módulos de código. El
identificador depende del contenido; repetir el respaldo del mismo estado
reutiliza el archivo después de verificarlo.

No incluye `.env`, credenciales de Google, entornos Python, caché E5, MLflow,
modelos joblib, datos históricos de entrenamiento ni cuarentena. El paquete del
modelo se conserva por separado. Los datos reales se guardan en el bucket privado,
no en GitHub. Sólo se versionan este código, sus pruebas y la documentación.

## Condiciones de V1

El diario debe estar en `fase: completo` y referenciar un historial que coincida
con las fuentes actuales. Un comprobante local de BigQuery se conserva y valida
contra ese historial; el programa no consulta BigQuery ni confirma su estado actual.
No debe estar ejecutándose `run_pipeline`, `run_incremental` ni otro escritor.
La creación usa el mismo bloqueo local del incremental. `run_pipeline` no respeta
ese bloqueo: debe estar detenido. El plan sólo lee y no crea bloqueos.

Si aparece una publicación pendiente o un bloqueo, se conserva para revisión;
este programa no lo borra ni reintenta trabajos cloud. No confundir los errores
históricos documentados en un lote con una operación actual pendiente.

Se rechazan enlaces, archivos inesperados dentro de los árboles operativos,
capturas sin manifiesto, predicciones sin registro, rutas peligrosas y datos
alterados. V1 admite hasta 20.000 archivos, 64 MiB por archivo y 512 MiB sin comprimir.

## Uso en PowerShell, desde la raíz del proyecto

Primero ejecutar las pruebas del proyecto y ver el plan:

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q tests
& '.\.venv\Scripts\python.exe' -B -m src.operations.state_checkpoint
```

El plan informa conteos, tamaño, identificador y destino. No consulta Mercado
Público, MLflow, Google Cloud ni carga el modelo. Luego crear:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.operations.state_checkpoint --crear
```

Se genera `data/processed/checkpoints/estado-<identificador>.tar.gz`. Conserva el
SHA256 mostrado. Verificar el archivo comprueba rutas, inventario, tamaños y huellas:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.operations.state_checkpoint --verificar '<archivo.tar.gz>' --sha256 '<sha256>'
```

Probar la recuperación en una carpeta NUEVA, cuyo padre exista:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.operations.state_checkpoint --recuperar '<archivo.tar.gz>' --sha256 '<sha256>' --destino '.\data\processed\recuperacion_prueba_01'
```

Esto no sustituye los datos del proyecto. Extrae en staging, verifica relaciones,
reconstruye el inventario y comprueba la identidad histórica antes de entregar la
carpeta. Si el destino ya existe, se detiene. Incluye copias de los cuatro módulos
como referencia para validar el historial; NO es un checkout completo ejecutable.
No se debe ejecutar el pipeline dentro de esa carpeta de prueba.

La recuperación requiere los mismos bytes de esos módulos, incluidos los finales
de línea. Guarda el commit correspondiente: un checkout Linux que convierta CRLF
a LF puede necesitar preparación explícita antes de superar esta validación.
Esta prueba local no demuestra aún funcionamiento en Linux o Cloud Run.

## Subida y recuperación desde Cloud Storage

Después de una recuperación local exitosa, subir el archivo con nombre inmutable
al prefijo `estado/checkpoints/` del bucket `imp-mlops-584518467840`, guardando el
SHA256 como metadato y usando `--if-generation-match=0` para impedir reemplazos.
Anotar la generación devuelta por `gcloud storage objects describe`. Al descargar,
usar esa generación y comprobar el SHA256 conservado antes de recuperar.

La subida y descarga se hacen explícitamente con gcloud; este módulo no tiene
dependencias cloud. El SHA256 del manifiesto detecta alteraciones accidentales;
no es una firma digital. El origen confiable sigue siendo el bucket privado y la
referencia de generación/SHA256 conservada fuera del archivo.

V1 no crea un puntero `latest`, no decide cuál checkpoint es más reciente y no
resuelve actualizaciones concurrentes. Esas piezas, los checkpoints entre
segmentos y la integración del modelo portable corresponden al siguiente paso
antes de automatizar con Cloud Run Jobs y Scheduler.
