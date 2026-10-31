# Baseline del Proyecto 2 y demostración histórica

Candidato: modelo alineado de 299 ejemplos. Criterio 0.2.0; perfil 0.1.0. El modelo y el test se conservaron durante la evaluación.

## Resultado de septiembre

| Medida | Resultado |
|---|---:|
| Acuerdo con etiquetas | 49/60 (81,67 %) |
| F1 macro | 0,669 |
| Relevantes recuperados | 13/17 (76,47 %) |
| Casos etiquetados en 3 recuperados como 3 | 7/10 |
| Acuerdo ponderado | 92,35 % |
| F1 macro ponderado | 0,450 |

Solo una de las 60 etiquetas está confirmada con el usuario. Las métricas miden principalmente concordancia con propuestas asistidas. Los pesos de muestreo cambian la interpretación: el recall ponderado de 96,75 % no sustituye la observación de 13/17 en esta muestra. La cobertura del archivo de septiembre va del 1 al 25; no se supone cobertura completa del mes.

## Errores y límites conservados

- El paquete de cursos de Lota incluye análisis de datos en salud y fue predicho en 0: una omisión importante.
- El curso explícito de Power BI y el soporte Informix se predijeron en 2 en lugar de sus etiquetas 3.
- Algunas etiquetas originales requieren futura auditoría: equipos TIC etiquetados en 0 frente a criterios posteriores de relación TIC débil; renovación y administración Tableau en 2 frente al componente de administración BI. No se cambiaron para mejorar este resultado.
- La mayor dificultad sigue estando entre 1, 2 y 3. La demo presenta el ranking como apoyo a la revisión, conservando errores y diferencias entre predicción y revisión.

## Demo local

La demo contiene los mismos 893 casos nuevos del piloto, con buscador, filtros, detalle, la referencia de 199 y el candidato de 299, seis revisiones confirmadas y 28 propuestas asistidas. También presenta la evaluación de septiembre. Es histórica y no consulta oportunidades abiertas en vivo. No ofrece información de organismo, región o cierre porque esos campos no están disponibles en este conjunto.

Para reproducirla, extraer este paquete en la raíz del proyecto y ejecutar:

```powershell
cd "D:\Python\Proyectos-Github\inteligencia-mercado-publico"
.\.venv\Scripts\python.exe -m src.reporting.build_historical_demo
```

Abrir data/processed/demo_mercado_publico.html en un navegador. La vista funciona sin un servidor ni conexión a una cuenta. Los textos originales se insertan como datos y se muestran con textContent; no se ejecutan como instrucciones o HTML.

## Trazabilidad

Hash del modelo: 371f8222a53f0fa355a3b19826174daa8772bc42c6b87312b95ad4856918185c

Hash del test: a79a5a7737034addd4cc9493e75cb9f8dee45c909a67120b48373c061d5598de

No se reentrenó ni se modificaron archivos del repositorio al generar la demo. Una nueva mejora requiere conservar este resultado como baseline y usar una evaluación posterior o independiente.

## Continuación del roadmap

Después de revisar la demo histórica: preparar ejecución reproducible del pipeline y registrar versiones y métricas; continuar con MLflow, BigQuery, Looker Studio y la ejecución programada en cloud. La limpieza del repositorio conserva resultados y experimentos de referencia.

## Enlaces oficiales y tabla para Looker Studio

Cada una de las 893 filas tiene un enlace a la ficha oficial construido con su CodigoExterno. Se muestra en el ranking y en el detalle, y abre una nueva pestaña. Si el portal no carga la ficha, se conserva el código para buscarla manualmente. No se inicia sesión ni se postula desde la demo.

Se comprobó el formato del enlace con las fichas oficiales 2258-299-LE25 y 1049-50-LE26 el 8 de octubre de 2026; no se verificó la disponibilidad individual de las 893 fichas. La construcción del parámetro se validó para todos los códigos. El estado mostrado en el HTML y en EstadoHistorico corresponde a la muestra histórica; el enlace permite consultar el estado vigente en el portal.

El mismo comando genera data/processed/oportunidades_looker.csv. Incluye códigos, títulos, publicación, estado histórico, posiciones y afinidades de ambos modelos, revisiones y UrlMercadoPublico. Tiene una fila por licitación y mantiene separados los resultados del modelo y las revisiones. No requiere reentrenar.

Para una primera vista en Looker Studio, usar UrlMercadoPublico como campo URL en una tabla, con CodigoExterno, Nombre, Afinidad299 y Posicion299. Ordenar por Posicion299 ascendente; incluir filtros de afinidad y búsqueda. La continuidad del proyecto mantiene BigQuery como fuente prevista para el tablero automatizado. Este CSV es una exportación histórica, no un tablero publicado ni una fuente actualizada.

La lógica de filtros y enlaces se comprobó con un DOM simulado. La validación visual de esta revisión queda en el navegador del usuario.
