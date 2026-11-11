# Preparación de una carpeta de trabajo desde Cloud Storage

## Para qué sirve

Conecta los dos paquetes que ya guardamos en el bucket: modelo y estado.
Descarga sus generaciones fijadas, comprueba tamaño/SHA256 y recupera el estado
en una carpeta separada. Copia allí el código del checkout actual y valida que
ese workspace puede utilizar el modelo portable.

No ejecuta capturas, inferencia, entrenamiento ni publicaciones. No crea servicios,
cuentas, permisos, buckets o tareas programadas. Sólo lee dos objetos de GCS.

## Referencias de esta entrega

El modelo conserva su referencia existente en config/cloud/model_bundle.json.
La referencia del estado se actualiza en config/cloud/state_checkpoint.json:

- checkpoint: b2f575f69da3398ef5ea33bf9c7a3a8b9d23fb1fd5986db65a662ec35569cda1
- generación: 1794353267618331
- SHA256: d2a6742a231220c7ce2e2d3b774f4db060f0ad7df8e50edf6ed147c4cadcbf35
- tamaño comprimido: 633076 bytes

Ese checkpoint corresponde al código de 4a7174b. Se conservan los objetos
anteriores del bucket; no se reemplazan ni se reinterpretan con el código nuevo.

## Paso a paso en PowerShell, desde la raíz del proyecto

Pruebas de esta entrega (sólo datos sintéticos):

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q tests/test_bootstrap_cloud.py
```

Ver el plan, sin red, búsqueda de credenciales o escritura de archivos:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.operations.bootstrap_cloud
```

Si el plan muestra el modelo y checkpoint esperados, preparar la carpeta:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.operations.bootstrap_cloud --preparar
```

Este último comando SÍ contacta Google Cloud y descarga aproximadamente 276,6 MiB.
Solicita las generaciones exactas; no busca un objeto llamado latest ni utiliza
los paquetes locales como alternativa si falla la descarga.

Usa ADC de Python, ya configurado durante la integración BigQuery. Las bibliotecas
necesarias son google-auth y requests, ya presentes en el entorno revisado
(google-auth 2.61.0). No invoca gcloud ni necesita una instalación nueva. En un
futuro trabajo Cloud Run, ADC usará la identidad del servicio, sin claves JSON.

La carpeta predeterminada es:

```text
data/processed/arranque_cloud_prueba_01/
  bootstrap.json
  descargas/
    modelo.tar.gz
    estado.tar.gz
  workspace/
    src/
    config/
    data/raw/batches/...
    data/processed/batches/...
    data/processed/incremental/...
    data/processed/history/...
    data/processed/reviews/...
    data/labeled/.../criterio.md
    sql/bigquery/...
```

El contenido de sql depende de los archivos del checkout. No copia .env, .git,
entornos virtuales, credenciales, MLflow local o la caché E5. El código copiado son
archivos .py de src; la configuración adicional son las dos referencias cloud y
el esquema BigQuery. El perfil y criterio se conservan desde el checkpoint.

Al finalizar debe indicar `estado: preparado`, mostrar el workspace, la ruta del
modelo local y los conteos recuperados: 10 lotes, 63 capturas, 60 predicciones,
1 revisión y 62 licitaciones en vista_actual. Esto no significa que haya nuevas
predicciones o datos publicados.

## Comprobaciones

- Generación en la solicitud y, cuando viene informado, en la respuesta.
- Tamaño y SHA256 de cada descarga; miembros y huellas internas del checkpoint.
- Identidad, cantidad de archivos y conteos del estado contra su referencia.
- Igualdad del código exigido por el recuperador.
- Recuperación de datos, relaciones y manifiestos en staging.
- Referencia de modelo del checkpoint igual a la referencia actual.
- Validación del modelo portable y sus cinco módulos de inferencia sin cargarlo.
- El código y referencias del checkout no cambiaron durante la preparación.

Las descargas usan el método oficial Objects.get con alt=media y generation.
La sesión autenticada obtiene y renueva credenciales mediante google-auth.
No se publican registros de tokens ni contenidos de excepciones en bootstrap.json.

Referencias técnicas:

- https://docs.cloud.google.com/storage/docs/json_api/v1/objects/get
- https://google-auth.readthedocs.io/en/latest/reference/google.auth.transport.requests.html

## Interrupciones y destinos existentes

El destino debe ser NUEVO y su padre debe existir. Nunca mezcla con un arranque
previo ni sobrescribe la carpeta del proyecto. Una segunda ejecución contra el
mismo destino se detiene. Para otro intento elegir un nombre nuevo:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.operations.bootstrap_cloud --destino '.\data\processed\arranque_cloud_prueba_02' --preparar
```

Un fallo conserva el intento y su bootstrap.json para inspección. Una descarga
interrumpida puede conservar un .part, que no se usa como paquete válido. No se
reintentan escrituras ni se borra automáticamente un intento anterior. Se requiere
espacio libre para descargas y staging; no se carga el encoder en esta etapa.

## Qué falta después

El workspace preparado es una copia independiente y las dependencias se siguen
tomando del Python que ejecuta el comando. Todavía no es un contenedor Linux ni
una tarea productiva. Este módulo no cambia el directorio de tu terminal ni activa
MP_MODEL_BUNDLE en otros procesos.

El siguiente ejecutor deberá seleccionar ese workspace y modelo, ejecutar una
corrida acotada, confirmar BigQuery y persistir un checkpoint nuevo. Después se
integrarán coordinación entre trabajos, identidad con permisos limitados,
contenedor, programación, monitoreo y reentrenamiento. No lanzar todavía
trabajadores concurrentes sobre una misma referencia de avance.
