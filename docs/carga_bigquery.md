# Historial de Python → BigQuery

Este bloque añade el cargador que faltaba entre el historial local y las cinco
tablas ya creadas en `inteligencia-mercado-publico.mercado_publico` (`us-central1`).
Looker y la programación semanal quedan después de comprobar esta conexión.

## 1. Colocar los archivos

Extrae el contenido del ZIP en la **raíz del proyecto**:

`D:\Python\Proyectos-Github\inteligencia-mercado-publico`

Las carpetas `src`, `config`, `sql`, `tests` y `docs` deben combinarse con las
existentes. Los archivos de este bloque son nuevos. No debe quedar una carpeta
intermedia `cargador_bigquery`, ni debe extraerse en `data/processed`.

Abre la terminal PowerShell de VS Code en esa raíz:

```powershell
Set-Location 'D:\Python\Proyectos-Github\inteligencia-mercado-publico'
```

## 2. Revisar el plan, sin conexión a Google

```powershell
.\.venv\Scripts\python.exe -B -m src.warehouse.load_history --historial data/processed/history/5c6220a1b9668af0e718
```

Resultado esperado con este historial:

| Tabla | Filas a comparar |
|---|---:|
| lotes | 4 |
| capturas | 8 |
| lote_captura | 14 |
| predicciones | 5 |
| revisiones | 1 |

La vista local tiene 7 licitaciones. Se verifican 34 archivos: 26 fuentes,
6 CSV del historial, su resumen y el constructor. `escrituras_cloud: false`
significa que sólo se preparó el plan; no significa que Google esté vacío.

## 3. Instalar la biblioteca y autenticar tu Python

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-bigquery.txt
gcloud auth application-default login
gcloud auth application-default set-quota-project inteligencia-mercado-publico
```

La biblioteca oficial permite llamar a BigQuery desde Python. La versión queda
fijada en el archivo de dependencias separado; no se cambia el entorno de
entrenamiento a otra versión de Python.

El segundo comando abre el inicio de sesión de Google. Usa la cuenta con la que
creaste el proyecto. Esto configura **Application Default Credentials (ADC)**:
el mecanismo que la biblioteca utiliza para encontrar tu identidad local.
Estar conectado en el navegador o ejecutar solamente `gcloud auth login` no
sustituye esta configuración. Las credenciales quedan fuera del repositorio.
No hay que descargar una clave JSON de cuenta de servicio ni pegar tokens.

`gcloud` se encontró instalado en este equipo; su autenticación no se pudo
comprobar desde la sesión de trabajo, porque su carpeta de configuración no
tiene permiso de escritura para esta sesión. Ejecuta estos comandos desde tu
terminal habitual. Si no se reconoce `gcloud`, abre una terminal nueva en VS Code.

Si aparece un error de permisos, revisa el mensaje antes de cambiar IAM. El
cargador necesita ejecutar trabajos y leer/escribir/crear/eliminar tablas en
este dataset; no necesita cambiar la facturación ni administrar otros proyectos.

## 4. Hacer la primera carga desde Python

```powershell
.\.venv\Scripts\python.exe -B -m src.warehouse.load_history --historial data/processed/history/5c6220a1b9668af0e718 --aplicar
```

`--aplicar` es lo que habilita las llamadas a Google. El comando:

1. Comprueba de nuevo las fuentes y sus huellas.
2. Comprueba región y esquemas de las cinco tablas existentes.
3. Crea cinco tablas auxiliares exclusivas de esta ejecución y sube JSON con
   tipos explícitos. Las auxiliares tienen caducidad de 24 horas.
4. Valida las relaciones y ejecuta cinco `MERGE` dentro de una transacción.
   Si un ID ya existe, exige contenido idéntico; sólo inserta IDs nuevos.
5. Valida la integridad resultante y confirma la transacción.
6. Muestra las filas anteriores, posteriores e insertadas por tabla y elimina
   las auxiliares cuando el resultado de la transacción está confirmado.

**La primera prueba reutiliza los datos que ya subimos por SQL.** Si las tablas
siguen en ese estado, esperamos `Insertadas: 0` en las cinco. Eso demuestra que
Python se conectó y la carga no duplicó los registros. No esperamos más
licitaciones hasta procesar un lote nuevo.

Repite el mismo comando una vez, después de terminar el anterior. Debe mostrar
otra vez cero insertadas. En BigQuery puedes verificar las filas con:

```sql
SELECT CodigoExterno, Nombre, AfinidadPredicha, AfinidadHumana, EstadoRevision
FROM `inteligencia-mercado-publico.mercado_publico.vista_actual`
ORDER BY CodigoExterno;
```

Nueva Alameda debe conservar predicción 0 y revisión 3. Las dos capturas sin
predicción deben seguir presentes. Cargar datos no vuelve a ejecutar el modelo
ni convierte las revisiones en ejemplos de entrenamiento.

## Si falla una carga

El cargador imprime el ID de trabajo antes de iniciar la carga. Busca ese ID
en el historial de trabajos de BigQuery (`us-central1`). Un error de conexión
no prueba que el servidor haya fallado: puede haberse confirmado la transacción
sin que el cliente recibiera la respuesta. Comprueba su estado antes de repetir.

Los errores durante la transacción revierten las incorporaciones de esa
transacción. Las auxiliares se conservan para diagnóstico en un fallo o resultado
desconocido y caducan solas en 24 horas. Si sólo falla su limpieza después de una
carga confirmada, el informe las enumera sin presentar la carga como fallida.

No ejecutes dos cargas al mismo tiempo. Esta versión garantiza la repetición
secuencial; todavía no implementa un bloqueo distribuido. Resolveremos la
coordinación antes de programar ejecuciones y reintentos cloud.

## Cómo se conecta con el siguiente lote

Después de capturar y predecir un lote nuevo, ejecutaremos el constructor de
historial que ya existe y pasaremos al cargador la **nueva carpeta que imprima**:

```powershell
.\.venv\Scripts\python.exe -B -m src.reporting.build_batch_history
```

El historial es acumulativo: se comparan sus eventos con los de BigQuery y se
insertan los nuevos. La vista SQL elige la captura más reciente y sus
predicción y revisión asociadas; no hay que subir `vista_actual.csv` ni
reemplazar la vista cada semana.

En Cloud Run usaremos una cuenta de servicio adjunta y ADC, con almacenamiento
persistente para capturas/modelos. Esta versión aún lee archivos locales;
el adaptador de Cloud Storage y la programación semanal no están implementados.

## Qué revisar del código para entenderlo

- `prepare_history`: prueba que estamos enviando datos íntegros y tipados.
- `transaction_sql`: conserva el comportamiento de nuestro SQL inicial sin
  incluir los valores de los datos dentro de un script SQL nuevo.
- `apply_history`: conecta, carga auxiliares y espera confirmación del servidor.
- `main`: conecta los argumentos de terminal con esas funciones.

Una tabla auxiliar es una zona de preparación. Una transacción permite
confirmar las cinco incorporaciones juntas. Una carga repetible mantiene los
mismos resultados al enviarse de nuevo, sin agregar duplicados.

## Verificación realizada antes de entregar

29 pruebas locales pasaron. Cubren huellas cambiadas, rutas fuera del proyecto,
tipos, ceros y vacíos, relaciones, probabilidades, tablas opcionales vacías,
fallos de carga, caducidad y limpieza. El cliente de Google se sustituye por un
doble para probar el protocolo: **esas pruebas no ejecutan SQL en BigQuery**.
También se validó el plan del historial real de 32 registros y 7 licitaciones,
sin modificar sus fuentes.

La instalación de la biblioteca, la autenticación, el SQL generado y la prueba
real de repetición en Google quedan pendientes de ejecutar en tu terminal.

Referencias oficiales:

- [Cliente Python de BigQuery](https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.client.Client).
- [ADC para desarrollo local](https://docs.cloud.google.com/docs/authentication/set-up-adc-local-dev-environment).
- [Transacciones de BigQuery](https://docs.cloud.google.com/bigquery/docs/transactions).
