# Preparar el modelo para ejecutarlo fuera del PC

Esta etapa exporta el candidato existente como un paquete con rutas relativas.
No reentrena, no captura nuevas licitaciones ni modifica el pipeline actual.
La adaptación del pipeline para consumirlo y la persistencia de las capturas
son etapas posteriores; todavía no se despliega Cloud Run.

## Qué contiene

- El joblib original del candidato de 299 ejemplos, comprobado contra el SHA256
  registrado en MLflow y en la predicción de referencia.
- Los diez archivos locales de E5 en la revisión
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- El perfil de empresa y el criterio de afinidad, contrastados con sus huellas
  históricas.
- Cinco textos de licitaciones con las clases y probabilidades históricas para
  una prueba de regresión.
- Un manifiesto de versiones, representación, fuentes y huellas de archivos.

El archivo contiene catorce archivos de datos, el manifiesto y los directorios
de módulos del encoder. No incluye `.env`, credenciales, SQLite de MLflow ni los
datasets de entrenamiento. Los hashes prueban integridad; la prueba numérica
comprueba la equivalencia con la referencia, no la calidad general del modelo.

## Instalación manual

Copiar al proyecto los archivos nuevos, respetando sus carpetas:

1. `src/experiments/export_model_bundle.py`
2. `src/experiments/verify_model_bundle.py`
3. `tests/test_model_bundle.py`
4. `docs/modelo_portable.md`

No se reemplazan archivos existentes del pipeline.

## 1. Pruebas

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q tests
```

Las pruebas nuevas usan archivos ficticios, no deserializan el clasificador ni
cargan el encoder y no contactan cloud. Cubren integridad, cambios en las fuentes,
reutilización, selección de archivos y rechazo de rutas/enlaces no admitidos.

## 2. Revisar el plan

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.experiments.export_model_bundle
```

Este comando es de lectura: verifica hashes del modelo, configuraciones y código
de inferencia, y prepara las cinco referencias. No importa bibliotecas de ML, no
abre el joblib y no descarga E5. Sin `--crear` no escribe el paquete.

La revisión de los archivos existentes encontró 14 archivos, unos 470,48 MiB sin
comprimir, el modelo versión 1 y el encoder de 384 dimensiones. Si falta la caché
local del encoder, se detiene; se puede indicar su ubicación con `--encoder-dir`.

## 3. Crear el paquete local

Después de revisar las pruebas y el plan:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.experiments.export_model_bundle --crear
```

Genera `data/processed/deployment/modelo-v1-<identificador>.tar.gz`. Puede tardar
por la compresión y las verificaciones de los pesos. Necesita espacio temporal
de aproximadamente 1 GiB en el disco de destino. Al terminar muestra la ruta,
el tamaño comprimido y el SHA256 del archivo.

El identificador depende de los archivos y metadatos. Repetir la exportación de
las mismas fuentes reutiliza el archivo, después de verificarlo. Un archivo ya
existente pero inválido no se sobreescribe. Si cambió alguna fuente, se detiene
o genera otro identificador, según el contrato verificado.

## 4. Comprobar que carga y reproduce la referencia

Usar la ruta exacta que imprima la exportación. Con las fuentes revisadas:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.experiments.verify_model_bundle --archivo '.\data\processed\deployment\modelo-v1-fa2a8793d037ded3.tar.gz'
```

La verificación extrae temporalmente el paquete dentro de
`data/processed/deployment`, comprueba los archivos y abre sólo el joblib cuyo
hash corresponde al candidato aprobado. Requiere las versiones registradas de
las bibliotecas de inferencia. Usa CPU y archivos locales con Hugging Face y
Transformers en modo offline. No utiliza MLflow, API ni BigQuery.

Compara las cinco clases exactamente y las probabilidades con las tolerancias
`rtol=1e-5`, `atol=1e-6`. También comprueba la longitud real de los fragmentos.
Al terminar retira sólo la carpeta temporal que creó; conserva el paquete.
Necesita espacio adicional para extraer el encoder durante esta comprobación.

Un aviso sobre la longitud del texto completo puede aparecer al tokenizar; la
verificación comprueba los fragmentos que realmente se representan y se detiene
si no respetan el límite.

## 5. Subir el paquete verificado al bucket privado

Sólo después de que la verificación numérica termine correctamente. Ajustar la
ruta si la exportación imprimió otro nombre:

```powershell
$archivoModeloMP = '.\data\processed\deployment\modelo-v1-fa2a8793d037ded3.tar.gz'
$nombreModeloMP = Split-Path -Leaf $archivoModeloMP
$hashModeloMP = (Get-FileHash -LiteralPath $archivoModeloMP -Algorithm SHA256).Hash.ToLowerInvariant()

gcloud.cmd storage cp $archivoModeloMP "gs://imp-mlops-584518467840/modelos/$nombreModeloMP" --project=inteligencia-mercado-publico --if-generation-match=0 --custom-metadata="sha256=$hashModeloMP"
```

La precondición evita reemplazar un objeto ya existente. Si se repite la subida
y aparece un error de precondición, revisar el objeto existente antes de cambiar
el nombre o sobrescribirlo. El SHA256 guardado como metadato sirve para comparar
el archivo cuando lo descarguemos; no sustituye la comprobación del contenido.

Referencia del comando:
https://docs.cloud.google.com/sdk/gcloud/reference/storage/cp

## Después de esta etapa

Se adaptará la inferencia para leer este paquete versionado, conservando las
identidades del modelo y los controles de regresión. También falta trasladar el
estado y las capturas, empaquetar la aplicación con dependencias para Linux y
configurar permisos, ejecución y programación. La prueba aquí se realiza en el
entorno actual; la compatibilidad del contenedor Linux se verificará al construirlo.

## Validación realizada al preparar la entrega

Se comprobó la sintaxis de los tres archivos Python y el plan de lectura con
las fuentes reales. Además se hicieron seis comprobaciones aisladas en la
carpeta de entrega, con archivos ficticios: plan, creación, reutilización,
identidad esperada, rutas fuera del destino y enlaces no admitidos.

No se ejecutó el modelo real ni se creó su archivo de 470 MiB para preparar la
entrega. Las pruebas completas del proyecto, la exportación real, la regresión
con E5 y la subida las ejecuta el usuario.
