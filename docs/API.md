# API

Servicio HTTP para alimentar y entrenar modelos de visión sobre planos arquitectónicos. Tiene
dos grupos de endpoints: **ingesta** (llegan planos, salen trazos coloreados) y
**entrenamiento** (ajuste del filtro y dataset coloreado para D-FINE, con versiones).

- Base: `http://<host>:8000`
- Documentación interactiva generada por FastAPI: `/docs` (Swagger) y `/redoc`
- Errores: código HTTP y cuerpo `{"detail": "mensaje"}`
- Autenticación: no incluida; se espera un gateway o proxy delante del servicio

## Ingesta

### POST /v1/planos

Sube un plano y lo encola para el filtro. Cuerpo `multipart/form-data` con el campo `file`.

| Formato | Detalle |
|---|---|
| PDF | Cada página se procesa por separado, rasterizada a `PDF_DPI` (200 por defecto) |
| TIFF | Todas las páginas |
| JPEG, PNG, WebP, BMP | Una página; se corrige rotación EXIF y transparencia |

El formato se detecta por el contenido, no por la extensión. Límites: `MAX_UPLOAD_MB` (100) por
archivo y `MAX_PIXELS` (150 millones) por página.

```bash
curl -F "file=@lamina.pdf" http://localhost:8000/v1/planos
```

Respuesta `202` (nuevo) o `200` (el mismo archivo ya se había subido):

```json
{"id": "9b1c4e5f0a2d4c3b8e7f6a5b4c3d2e1f", "estado": "en_cola", "duplicado": false}
```

Con cola Redis el estado inicial es `en_cola`; en modo local (cola en línea) la respuesta llega
ya con `listo` o `error`. Errores: `413` archivo muy grande, `415` formato no soportado.

### GET /v1/planos/{id}

```json
{
  "id": "9b1c4e5f0a2d4c3b8e7f6a5b4c3d2e1f",
  "sha256": "5e8a...",
  "nombre": "lamina.pdf",
  "formato": "pdf",
  "bytes": 482113,
  "clave_original": "originales/5e8a.../lamina.pdf",
  "estado": "listo",
  "error": null,
  "creado": "2026-10-22T21:10:04+00:00",
  "paginas": [
    {
      "numero": 1,
      "estado": "ok",
      "salidas": {
        "color": "procesado/9b1c.../p1/color.png",
        "etiquetas": "procesado/9b1c.../p1/etiquetas.png",
        "hints": "procesado/9b1c.../p1/hints.png"
      },
      "metricas": {
        "status": "ok",
        "wall_thickness_px": 14.0,
        "footprint_frac": 0.2066,
        "wall_px": 262103,
        "glass_px": 9001,
        "n_buildings": 1,
        "footprint_wkt": "MULTIPOLYGON (((...)))",
        "n_detections": 0,
        "processing_scale": 1.0,
        "seconds": 0.326
      }
    }
  ]
}
```

Estado del plano: `en_cola`, `procesando`, `listo`, `error`. Estado de cada página:

| Estado | Significado | Qué hacer en el funnel |
|---|---|---|
| `ok` | Límite encontrado | Usar |
| `cropped` | El edificio toca el borde de la imagen; el resultado suele ser peor | Revisar o descartar |
| `no_footprint` | No se encontró un edificio | Descartar o revisar |
| `unbounded` | El relleno no pudo entrar (marco o plano que llena la lámina) | Revisar |
| `too_dark` | Más de 35 % de tinta (fondo oscuro o rellenos fuertes) | Descartar |

### GET /v1/planos/{id}/paginas/{n}/{salida}

Devuelve un PNG de la página `n`.

| Salida | Contenido | Para qué |
|---|---|---|
| `color` | El plano con los trazos del límite coloreados: rosado muro, azul vidrio. El grosor del dibujo no cambia | Entrada de D-FINE y revisión visual |
| `etiquetas` | Un canal: 0 fondo, 1 muro, 2 vidrio | Entrenar segmentadores de trazos |
| `hints` | Tres canales: B plano original, G muro, R vidrio | Modelos que prefieran las clases en canales |

### GET /v1/parametros

Parámetros con los que trabaja el filtro y su versión (`config` si vienen del archivo).

```json
{"version": "20261022-214500", "parametros": {"max_side": 1600, "faint_line_contrast": 30, "bridge_factor": 10.0, "...": "..."}}
```

## Entrenamiento

Los entrenamientos corren en el trainer, con su propia cola, para no frenar la ingesta. Cada
resultado queda como una versión en el registro de modelos y se activa por separado.

### POST /v1/entrenamientos

```json
{
  "tipo": "ajuste_parametros",
  "parametros": {"iteraciones": 24, "planos_por_estilo": 80, "semilla": 0, "medicion_por_estilo": null},
  "activar": false
}
```

| Tipo | Qué hace | Parámetros |
|---|---|---|
| `ajuste_parametros` | Partición 30/70 por estilo de dibujo, iteraciones sobre el 30 %, medición en el 70 %. Registra una versión de parámetros del filtro | `iteraciones` (24), `planos_por_estilo` (80), `semilla` (0), `medicion_por_estilo` (todos) |
| `dataset_dfine` | Pasa los planos anotados por el filtro activo y escribe imágenes coloreadas con cajas COCO de ventanas y puertas. 30 % train, 70 % val | `planos_por_estilo` (todos), `semilla` (0) |
| `segmentador_trazos` | Flujo B. Responde `501` hasta decidir el modelo base | |

Respuesta `202`:

```json
{"id": "c47e0b3a9d1f4e2a8b6c5d4e3f2a1b0c", "tipo": "ajuste_parametros", "estado": "en_cola"}
```

Errores: `422` tipo desconocido, `501` tipo pendiente.

### GET /v1/entrenamientos/{id}

Ejemplo de un ajuste terminado (valores del ajuste real de este repositorio):

```json
{
  "id": "c47e0b3a9d1f4e2a8b6c5d4e3f2a1b0c",
  "tipo": "ajuste_parametros",
  "parametros": {"iteraciones": 24, "planos_por_estilo": 80},
  "estado": "listo",
  "metricas": {
    "mejor_iteracion": 12,
    "iteraciones": 24,
    "f1_balanceado_30": 0.789,
    "medicion_70": {
      "actuales": {"colorful": 0.747, "cvc_fp": 0.815, "high_quality": 0.774, "high_quality_architectural": 0.614, "f1_balanceado": 0.738, "f1_global": 0.662, "planos": 3584},
      "mejores": {"colorful": 0.775, "cvc_fp": 0.865, "high_quality": 0.816, "high_quality_architectural": 0.675, "f1_balanceado": 0.783, "f1_global": 0.723, "planos": 3584}
    }
  },
  "artefactos": {"modelo_id": "5d2e...", "version": "20261022-214500", "uri": "modelos/parametros_filtro/20261022-214500.json"},
  "error": null
}
```

Un `dataset_dfine` terminado:

```json
{
  "tipo": "dataset_dfine",
  "estado": "listo",
  "metricas": {
    "prefijo": "datasets/dfine/20261022-230000",
    "imagenes": {"train": 1538, "val": 3584},
    "cajas": {"train": 28000, "val": 65000},
    "estados_filtro": {"ok": 4700, "cropped": 400, "no_footprint": 22}
  },
  "artefactos": {"modelo_id": "8a1f...", "version": "20261022-230000", "prefijo": "datasets/dfine/20261022-230000"}
}
```

(Las cifras de este segundo ejemplo son ilustrativas.) Estado: `en_cola`, `corriendo`, `listo`, `error`.

### GET /v1/modelos

Versiones registradas, la más reciente primero. Filtro opcional `?tipo=parametros_filtro`,
`dataset_dfine`, `segmentador_trazos` o `dfine`.

```json
[
  {"id": "5d2e...", "tipo": "parametros_filtro", "version": "20261022-214500",
   "uri": "modelos/parametros_filtro/20261022-214500.json", "metricas": {"...": "..."},
   "activo": true, "creado": "2026-10-22T21:45:00+00:00"}
]
```

### POST /v1/modelos/{id}/activar

Deja esa versión como la única activa de su tipo. Los workers la toman en el siguiente plano,
sin reiniciar. Devuelve el modelo con `"activo": true`; `404` si no existe.

## Operación

| Endpoint | Respuesta |
|---|---|
| `GET /health` | `{"estado": "ok", "version": "0.2.0"}` |
| `GET /ready` | `{"storage": true, "cola": true, "cola_entrenamiento": true, "repositorio": true}`; `503` si alguno es `false` |

## Mensajes en las colas

Redis Streams, un campo `job` con JSON. Ingesta en `QUEUE_STREAM`, entrenamiento en
`TRAINING_STREAM`; los que fallan quedan en `<stream>:fallidos` con el error.

```json
{"kind": "procesar_plano", "payload": {"plan_id": "9b1c...", "raw_key": "originales/5e8a.../lamina.pdf"}, "id": "f3a2..."}
{"kind": "ajuste_parametros", "payload": {"training_id": "c47e...", "parametros": {"iteraciones": 24}, "activar": false}, "id": "a1b2..."}
```

## Usar el dataset coloreado con D-FINE

Estructura bajo el prefijo de la versión:

```
images/train/<plano>.png
images/val/<plano>.png
annotations/train.json      COCO, categorías 1 window y 2 door
annotations/val.json
```

En D-FINE (`configs/dataset/custom_detection.yml`): `num_classes: 2`,
`remap_mscoco_category: False`, `img_folder` y `ann_file` apuntando a esas rutas. Usar pesos
preentrenados solo en COCO y subir la resolución de entrada (640 por defecto) o trabajar por
mosaicos, para que el vidrio coloreado no se aclare al reducir.
