# Registro local de la baseline en MLflow

## Objetivo

Importar el candidato affinity_aligned_299 ya guardado, los resultados de validación de agosto y la evaluación de septiembre. Se registra como importación histórica: el inicio del run es la fecha de importación y no la fecha de entrenamiento. No se entrena, no se descarga el encoder y no se calculan predicciones ni métricas nuevas. No se registra una evaluación de septiembre del modelo 199.

## Instalación

Descomprimir el paquete y combinar sus carpetas src y docs con las existentes en la raíz del proyecto. Dejar requirements-mlflow.txt y constraints-mlflow.txt también en la raíz. No reemplazar requirements.txt ni borrar las carpetas existentes.

Desde la terminal de PowerShell:

```powershell
cd "D:\Python\Proyectos-Github\inteligencia-mercado-publico"
.\.venv\Scripts\python.exe -m pip install -r requirements-mlflow.txt -c constraints-mlflow.txt
```

MLflow está fijado en 3.16.1. Las restricciones conservan las versiones de las bibliotecas de datos y modelo observadas en el entorno de entrenamiento. Si la instalación informa un conflicto, conservar el mensaje para revisarlo antes de cambiar estas versiones.

## Importar y abrir

```powershell
.\.venv\Scripts\python.exe -m src.experiments.register_affinity_baseline
.\.venv\Scripts\python.exe -m src.experiments.open_mlflow
```

Abrir http://127.0.0.1:5000 en el navegador. La segunda terminal queda ocupada mientras el servidor está funcionando. Ctrl+C lo detiene; se puede abrir de nuevo sin repetir la importación. Si el puerto está ocupado, ejecutar open_mlflow con --port 5001 y abrir la dirección indicada.

En Experiments buscar "Proyecto 2 - Afinidad Mercado Publico" y el run "baseline_299_importacion_historica". Revisar Params, Metrics y Artifacts. Las métricas comienzan con validation_ o test_ para distinguir períodos. En Models buscar "mercado_publico_afinidad_embeddings": la primera importación crea una versión y el alias candidato. Ese alias identifica el candidato para la siguiente integración, no una aprobación de producción.

## Qué se guarda

- Parámetros del clasificador, encoder previsto, versiones de perfil y criterio, cantidades y meses de cada conjunto.
- Métricas de los reportes existentes; para septiembre Exactitud 0.816667 y F1Macro 0.668595. Las etiquetas son mayoritariamente asistidas.
- El joblib original, copias de los conjuntos y archivos congelados, reportes, código de referencia y trazabilidad de los archivos. No se incluye .env ni credenciales.
- Una copia del clasificador en formato MLflow sklearn, con firma de entrada float32 de 384 dimensiones y salida afinidad 0–3.

El clasificador registrado recibe embeddings, no textos crudos. La preparación de textos, fragmentación, encoder multilingual-e5-small y promedio normalizado continúan en el código del proyecto; la versión del encoder se identifica, pero sus pesos no se empaquetan ni descargan en este paso. El registro del clasificador no despliega un servicio completo de licitaciones.

## Archivos locales

La base está en mlflow_local/mlflow.db y los artefactos en mlflow_local/artifacts. Mantener ambos juntos y hacer las copias de seguridad con el servidor detenido. El comprobante de la importación está en data/processed/affinity_mlflow_baseline_registration.json e incluye run, versión, URI y hash del modelo. La ubicación absoluta de los artefactos se usa para este registro local; trasladarlo a otra máquina requiere migrar o reimportar el registro.

Añadir mlflow_local/ al .gitignore del proyecto antes de un futuro commit: la base y los artefactos son resultados locales. Conservar requirements-mlflow.txt y constraints-mlflow.txt como dependencias de este paso. No se reemplaza ni edita el modelo original.

Repetir register_affinity_baseline reutiliza un run finalizado con la misma identidad en vez de duplicarlo. La opción --check verifica los archivos sin escribir y sin necesitar MLflow. No ejecutar importaciones simultáneas: la comprobación de duplicados está diseñada para uso local secuencial. Un run fallido se conserva para diagnóstico.

## Comprobaciones del paquete

Se verificó la coincidencia del resultado histórico y sus archivos contra el repositorio real mediante --check. Las pruebas locales de contrato usan MLflow simulado para comprobar separación de métricas, artefactos, repetición y rechazo de archivos alterados. La instalación de MLflow, creación real del registro y vista en navegador se confirman al ejecutar los pasos anteriores en el entorno del usuario.

## Fuentes técnicas

- https://mlflow.org/docs/latest/ml/tracking/tracking-api
- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html
- https://mlflow.org/docs/latest/api_reference/cli.html
- https://pypi.org/project/mlflow/3.16.1/

Una vez confirmado el registro, el siguiente bloque conecta la API de Mercado Público con la preparación de textos y el candidato guardado; luego continúa BigQuery, Looker Studio y la programación en cloud.
