# Diccionario inicial de datos

Este documento describe los campos iniciales que se utilizarán en el MVP de
Inteligencia de Oportunidades en Mercado Público.

El diccionario no incluye todos los campos entregados por la API. Se priorizan
aquellos necesarios para identificar, filtrar, ordenar y explicar oportunidades
relevantes para una consultora de datos.

## Datos de la licitación

| Campo de origen | Tipo observado | Nombre normalizado | Uso previsto |
|---|---|---|---|
| `CodigoExterno` | string | `tender_id` | Identificador único de la licitación |
| `Nombre` | string | `tender_name` | Texto principal para búsqueda y NLP |
| `Descripcion` | string | `tender_description` | Descripción general para NLP |
| `CodigoEstado` | integer | `status_code` | Código del estado |
| `Estado` | string | `status_name` | Estado interpretable de la licitación |
| `Tipo` | string | `tender_type` | Tipo de proceso de compra |
| `Moneda` | string | `currency` | Moneda del proceso |
| `MontoEstimado` | numérico o nulo | `estimated_amount` | Valor comercial estimado |
| `Informada` | integer | `amount_is_informed` | Indica si el monto fue informado |

## Organismo comprador y territorio

| Campo de origen | Tipo observado | Nombre normalizado | Uso previsto |
|---|---|---|---|
| `Comprador.NombreOrganismo` | string | `buyer_organization` | Organismo responsable |
| `Comprador.NombreUnidad` | string | `buyer_unit` | Unidad compradora |
| `Comprador.ComunaUnidad` | string | `buyer_commune` | Señal territorial |
| `Comprador.RegionUnidad` | string | `buyer_region` | Prioridad para Los Ríos y Los Lagos |

Los datos personales del usuario comprador, como nombre, cargo o correo, no
formarán parte del dataset analítico inicial porque no son necesarios para el
ranking de relevancia.

## Fechas

| Campo de origen | Tipo observado | Nombre normalizado | Uso previsto |
|---|---|---|---|
| `Fechas.FechaCreacion` | string | `created_at` | Fecha de creación |
| `Fechas.FechaPublicacion` | string | `published_at` | Fecha de publicación |
| `Fechas.FechaCierre` | string | `closing_at` | Fecha límite para participar |
| `Fechas.FechaAdjudicacion` | string o nulo | `awarded_at` | Análisis histórico |
| `Fechas.FechaEstimadaAdjudicacion` | string o nulo | `estimated_award_at` | Planificación del proceso |

Las fechas se convertirán posteriormente desde texto a valores de fecha y hora.

## Productos y servicios solicitados

Cada licitación puede contener uno o más elementos en `Items.Listado`.

| Campo de origen | Tipo observado | Nombre normalizado | Uso previsto |
|---|---|---|---|
| `Items.Listado[].Correlativo` | integer | `item_number` | Identificador interno del ítem |
| `Items.Listado[].CodigoProducto` | integer | `product_code` | Código del producto o servicio |
| `Items.Listado[].CodigoCategoria` | string | `category_code` | Código de categoría |
| `Items.Listado[].Categoria` | string | `category_name` | Señal temática para relevancia |
| `Items.Listado[].NombreProducto` | string | `product_name` | Texto para NLP |
| `Items.Listado[].Descripcion` | string | `item_description` | Texto principal para NLP |
| `Items.Listado[].UnidadMedida` | string | `unit_of_measure` | Unidad solicitada |
| `Items.Listado[].Cantidad` | float | `quantity` | Cantidad solicitada |

## Información histórica de adjudicación

| Campo de origen | Tipo observado | Nombre normalizado | Uso previsto |
|---|---|---|---|
| `Adjudicacion.Fecha` | string o nulo | `award_date` | Análisis histórico |
| `Adjudicacion.NumeroOferentes` | integer o nulo | `number_of_bidders` | Análisis de competencia histórica |
| `Adjudicacion.UrlActa` | string o nulo | `award_document_url` | Trazabilidad hacia el acta |
| `Items.Listado[].Adjudicacion` | dictionary | `item_award` | Resultado histórico por ítem |

Los datos de adjudicación no se utilizarán para afirmar una “probabilidad de
ganar”. El objetivo del producto es construir un ranking explicable de
relevancia respecto del perfil empresarial.

## Campos generados por el sistema

| Campo generado | Tipo esperado | Uso previsto |
|---|---|---|
| `retrieved_at` | datetime | Momento de extracción desde la API |
| `source_date` | date | Fecha consultada en la API |
| `relevance_score` | float | Puntaje final de relevancia |
| `matched_terms` | list | Palabras o conceptos que explican el puntaje |
| `territorial_priority` | boolean | Prioridad para Los Ríos o Los Lagos |
| `ranking_version` | string | Versión de la lógica o modelo utilizado |