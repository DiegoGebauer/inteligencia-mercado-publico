# Cierre de auditoría del piloto

AUD002 queda en 2: IA aplicada a docencia es temática relacionada, sin un componente directo de BI, ingeniería o análisis de datos. El usuario confirmó las otras cinco propuestas del bloque.

Seis casos confirmados; los otros 28 conservan su origen asistido. La auditoría no se incorpora automáticamente al entrenamiento.

## Casos confirmados

| Caso | Afinidad | Motivo |
|---|---:|---|
| AUD001 | 1 | El texto solicita una plataforma de reclutamiento y evaluación de personas, sin un trabajo explícito de gestión o análisis de datos. La categoría transformación de datos no basta para asignar 3. |
| AUD002 | 2 | Afinidad 2 confirmada con el usuario: capacitación en herramientas de IA para docentes, relacionada con el área pero sin componente directo de BI, ingeniería o análisis de datos. |
| AUD005 | 1 | Cursos de salud, probidad y autocuidado con una línea de ciberseguridad. Esta aporta relación TIC débil; no hay capacitación directa en datos. |
| AUD012 | 1 | Cursos de OSINT para Red Team y ciberseguridad preventiva/defensiva. Relación TIC débil; seguridad queda fuera de la oferta principal. |
| AUD013 | 3 | Gestión de asistencia con automatización, trazabilidad, monitoreo y generación de reportes. Propuesta de afinidad directa por el componente de gestión de datos, con operación de RR. HH. evaluada aparte. |
| AUD021 | 3 | Medición y monitoreo del progreso lector con análisis de evolución y soporte a decisiones. Propuesta de afinidad directa por el componente analítico; aplicación pedagógica y operativa se evalúa aparte. |

## Selección del candidato inicial

Se selecciona el modelo alineado de 299 para la evaluación final de septiembre. Mejoró el F1 macro ponderado de agosto (0,609 frente a 0,496) y, en la lectura del top10, propone cinco oportunidades de relación directa en 3 frente a tres del modelo de 199. Estos juicios del piloto mezclan confirmaciones humanas y propuestas asistidas.

El modelo de 199 conserva mejor accuracy de agosto (90 % frente a 86,67 %) y siete resultados relacionados en 2 o 3 dentro del top10 frente a seis de 299; algunos son compras de licencias. La elección de 299 prioriza oportunidades de servicios directos para este prototipo y el criterio ponderado usado durante el desarrollo. No significa superioridad universal ni que el ranking carezca de errores. Por ejemplo, ambos modelos colocan demasiado arriba una plataforma genérica de reclutamiento.

Se conserva el modelo de 199 como referencia. La elección del candidato, los hashes y los motivos quedan congelados antes de calcular resultados de septiembre.

## Instalación y evaluación

Extraer el paquete en la raíz del proyecto conservando las carpetas src, data y docs. Ejecutar:

```powershell
cd "D:\Python\Proyectos-Github\inteligencia-mercado-publico"
.\.venv\Scripts\python.exe -m src.nlp.evaluate_aligned_test
```

El comando aplica el candidato guardado a los 60 casos de septiembre, sin reentrenar ni comparar candidatos allí. Guarda data/processed/affinity_aligned_299_test_report.json y affinity_aligned_299_test_predictions.csv. Si ya existen, detiene la ejecución para conservar esta evaluación.

Las etiquetas de septiembre siguen en su versión original, mayoritariamente asistida. La revisión de seis casos del piloto no valida todo el dataset ni cambia el test. Los resultados deberán presentarse con esa limitación. Una mejora posterior del modelo requerirá una nueva evaluación posterior o independiente, sin presentar septiembre como un test intacto otra vez.

Se verificaron el script, la selección y la lógica de evaluación con datos y clasificador simulados. La evaluación real de septiembre queda pendiente de ejecutar el comando.

## Auditoría completa

| Caso | Código | Afinidad propuesta o confirmada | Origen |
|---|---|---:|---|
| AUD001 | 2258-299-LE25 | 1 | revision_con_usuario |
| AUD002 | 439-21-LE26 | 2 | revision_con_usuario |
| AUD003 | 1049-50-LE26 | 3 | asistida |
| AUD004 | 1271359-66-LE26 | 0 | asistida |
| AUD005 | 2292-25-LE26 | 1 | revision_con_usuario |
| AUD006 | 622-30-LR26 | 3 | asistida |
| AUD007 | 2408-19-LE26 | 3 | asistida |
| AUD008 | 2258-14-LP26 | 2 | asistida |
| AUD009 | 770-3-LE26 | 2 | asistida |
| AUD010 | 519302-2-LE26 | 2 | asistida |
| AUD011 | 3047-26-LE26 | 0 | asistida |
| AUD012 | 4146-3-L126 | 1 | revision_con_usuario |
| AUD013 | 2793-90-LP26 | 3 | revision_con_usuario |
| AUD014 | 642028-2-LE26 | 3 | asistida |
| AUD015 | 707425-24-L126 | 0 | asistida |
| AUD016 | 622-4-LE26 | 1 | asistida |
| AUD017 | 2832-18-LE26 | 0 | asistida |
| AUD018 | 1057547-60-LR26 | 0 | asistida |
| AUD019 | 3260-14-L126 | 0 | asistida |
| AUD020 | 4776-44-LP26 | 0 | asistida |
| AUD021 | 552975-11-LE26 | 3 | revision_con_usuario |
| AUD022 | 621-580-LE26 | 0 | asistida |
| AUD023 | 4371-18-LE26 | 0 | asistida |
| AUD024 | 1469-57-L126 | 1 | asistida |
| AUD025 | 2793-65-LP26 | 0 | asistida |
| AUD026 | 2582-24-LP26 | 0 | asistida |
| AUD027 | 2905-13-LE26 | 0 | asistida |
| AUD028 | 4294-6-LR26 | 0 | asistida |
| AUD029 | 3882-34-LE26 | 0 | asistida |
| AUD030 | 2674-18-LE26 | 0 | asistida |
| AUD031 | 3191-62-LE26 | 0 | asistida |
| AUD032 | 4956-15-LP26 | 0 | asistida |
| AUD033 | 2981-107-LE26 | 0 | asistida |
| AUD034 | 2591-107-LE26 | 0 | asistida |
