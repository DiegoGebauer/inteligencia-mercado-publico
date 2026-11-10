# Inferencia del pipeline desde el paquete aprobado

## Resultado buscado

Los ejecutores existentes (`run_pipeline`, `run_incremental` y
`run_incremental_batch`) siguen usando `recent_batch.predecir_lote`. Esa función
ahora puede elegir un cargador portable cuando se define `MP_MODEL_BUNDLE`.
El modo portable usa el archivo local aprobado y su encoder empaquetado.

La selección es explícita. Sin esa variable se conserva la ruta anterior de
MLflow. Si está definida pero vacía, apunta a una URI o el paquete es inválido,
la ejecución falla: no cambia silenciosamente de modelo ni de cargador.

No se cambia el clasificador ni se reentrena E5. Se conservan la preparación del
texto, fragmentos de 400 tokens con solapamiento 40, prefijo `passage: `,
representación normalizada de 384 dimensiones y cálculo de probabilidades.

## Archivos y verificaciones

El descriptor `config/cloud/model_bundle.json` fija generación GCS, tamaño,
SHA256, identidad del paquete, modelo y encoder. Este cargador no consulta GCS:
comprueba que el archivo local coincide con esa referencia.

Antes de abrir el joblib comprueba su huella aprobada, miembros del archivo,
modelo, encoder, representación, perfil de empresa y huellas de los cinco
módulos de inferencia. Conserva las versiones exactas de las seis bibliotecas
registradas en el paquete. Una diferencia detiene la carga; no se actualizan
dependencias automáticamente.

Prepara una copia temporal verificada del paquete, extrae sólo los miembros
esperados y carga el encoder con ruta local, CPU y `local_files_only=True`.
El contexto activa el modo offline de Hugging Face y restaura las variables al
terminar. No requiere catálogo SQLite de MLflow, artefactos MLflow, el joblib
original fuera del paquete ni una caché E5 previa.

La generación del descriptor identifica el objeto de origen; validar un archivo
local no comprueba por sí mismo el estado actual del objeto en GCS. La descarga
por generación y posterior validación SHA256 se integrarán en el trabajador cloud.

## Prueba paso a paso

Con los ejecutores detenidos, ejecutar las pruebas desde la raíz del proyecto:

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q tests
```

Dejar `MP_MODEL_BUNDLE` sin definir durante las pruebas generales: los tests
anteriores de la ruta MLflow suponen el modo local original. Los tests nuevos
controlan su propia selección. Antes de definirla globalmente, ejecutar:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.inference.portable_model --archivo '.\data\processed\deployment\modelo-v1-fa2a8793d037ded3.tar.gz'
```

Este comando sólo lee y valida archivos: no importa las bibliotecas de ML, no
deserializa el modelo, no escribe archivos ni contacta servicios.

Después ejecutar la comparación a través de la ruta real del pipeline:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.experiments.verify_portable_pipeline --archivo '.\data\processed\deployment\modelo-v1-fa2a8793d037ded3.tar.gz'
```

Esta prueba SÍ carga el modelo en CPU y hace inferencia. Crea un proyecto temporal
dentro de `data/processed/deployment`, copia las cinco capturas de referencia y
los archivos de configuración/código necesarios, y usa `predecir_lote` con la
selección portable. No copia la base MLflow ni los modelos originales. Usa una
ubicación nueva para la caché HF y bloquea las funciones del cargador anterior.

Compara códigos, textos, clases exactas, probabilidades y afinidad esperada
con tolerancias rtol=1e-5 y atol=1e-6. Comprueba la trazabilidad y que el constructor
de historial acepta las predicciones. Verifica que no cambiaron las capturas y
resultados originales; elimina únicamente su carpeta temporal.

No consulta Mercado Público ni publica en BigQuery. Esta comprobación demuestra
equivalencia de ejecución, no calidad predictiva ni funcionamiento en Linux.

## Activación posterior en la terminal actual

Después de superar las comprobaciones:

```powershell
$env:MP_MODEL_BUNDLE = (Resolve-Path -LiteralPath '.\data\processed\deployment\modelo-v1-fa2a8793d037ded3.tar.gz').Path
```

Los comandos habituales de los tres ejecutores usarán el paquete al necesitar
una predicción nueva. Una predicción completa existente seguirá reutilizándose:
no prueba el cargador nuevo ni se recalcula sólo por cambiar esta variable.
En este hito ambos modos están fijados al mismo modelo v1; no se implementa
selección de versiones nuevas o promoción automática.

Para volver al modo anterior en esa terminal:

```powershell
Remove-Item Env:MP_MODEL_BUNDLE -ErrorAction SilentlyContinue
```

La comprobación del modelo se hace al iniciar una inferencia. Una captura nueva
puede precederla en los coordinadores: validar el paquete antes de iniciar una
corrida real evita descubrir un archivo ausente después de capturar datos.

## Registro de ejecución e historial

El CSV mantiene las columnas requeridas por el historial y BigQuery. La identidad
del modelo conserva URI, versión, run, SHA del clasificador, encoder, revisión,
perfil y criterio. El alias mostrado es `version_fija_portable`: este modo no
resuelve el alias candidato de un registro MLflow.

El JSON añade `origen_carga_modelo: paquete_portable` y `paquete_modelo` con la
referencia exacta, además de versiones de paquetes y huellas del código. No
incluye una versión MLflow como dependencia. Los intentos de inferencia fallidos
que ya empezaron se conservan; no se sobrescriben predicciones completas.

El checkpoint anterior queda asociado al código de su momento. Su recuperación
estricta seguirá requiriendo ese código (commit e797ebb); este cambio modifica
`recent_batch.py`. Antes del despliegue se generará otro checkpoint con la versión
validada del ejecutor. No sustituir ni editar el respaldo anterior para forzarlo.

## Alcance y siguiente etapa

Esta entrega conecta inferencia local y paquete portable. No descarga objetos,
no instala contenedores, no crea Cloud Run/Scheduler y no sincroniza estado entre
trabajadores. Siguen pendientes esas integraciones, Linux, monitoreo y el ciclo
de reentrenamiento/evaluación/promoción.
