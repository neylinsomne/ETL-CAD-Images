# Sistema

Proyecto de código abierto orientado a **modelos de visión** sobre planos arquitectónicos. El
filtro es un paso de ingesta: cada plano que llega (PDF o imagen) se convierte en páginas, en
cada página se busca el límite exterior del edificio y se colorean los trazos de ese límite,
rosado para el muro y azul para el vidrio (ventanas, muros cortina, vidrieras). Esas imágenes
coloreadas son la entrada con la que se entrenan y usan los modelos de visión, como D-FINE, que
no corre dentro de este paso.

Referencia de endpoints y formatos JSON: [API.md](API.md). Investigación de repositorios y
modelo base: [INVESTIGACION.md](INVESTIGACION.md).

## Componentes

```mermaid
flowchart TB
    subgraph Servicio
        API["API FastAPI"]
        W["Workers de ingesta"]
        T["Trainer"]
    end
    subgraph Externos["Infraestructura externa (se asume existente)"]
        R[["Redis Streams: ingesta y entrenamiento"]]
        PGB["PgBouncer"] --> PG[("Postgres + pgvector")]
        S3[("S3 o compatible")]
    end
    API --> R
    R --> W
    R --> T
    API --> PGB
    W --> PGB
    T --> PGB
    API --> S3
    W --> S3
    T --> S3
    T --> D[("Datasets anotados locales")]
```

| Componente | Qué hace | Código |
|---|---|---|
| API | Recibe planos, deduplica, encola, sirve resultados, lanza entrenamientos, activa versiones | `service/api.py` |
| Worker | Procesa planos página por página con la versión activa de parámetros | `service/worker.py` |
| Trainer | Ajusta parámetros y arma datasets para D-FINE, en su propia cola | `service/trainer.py`, `training/` |
| Cola | Redis Streams con grupo de consumidores y cola de fallidos | `infra/queue.py` |
| Almacenamiento | Originales, salidas, versiones de parámetros y datasets | `infra/storage.py` |
| Metadatos | Planos, páginas, entrenamientos y registro de modelos | `infra/repository.py`, `sql/001_init.sql` |
| Filtro | Límite exterior y coloreado de trazos | `envelope.py`, `strokes.py`, `pipeline.py` |

Redis, PgBouncer, Postgres con pgvector y S3 se asumen existentes: el código solo necesita sus
direcciones (`.env.example`). Para desarrollo y pruebas cada uno tiene reemplazo local: cola en
línea (procesa al recibir), disco y memoria.

## Flujo de ingesta

```mermaid
flowchart LR
    C["Cliente o proceso de ingesta"] -->|"POST /v1/planos"| API
    API -->|"original"| S3o[("S3 originales/")]
    API -->|"plano en_cola"| PG[("Postgres")]
    API -->|"procesar_plano"| R[["Redis: ingesta"]]
    R --> W["Worker"]
    S3o --> W
    W --> I["Formato a páginas"]
    I --> A["Flujo A: límite por reglas"]
    B["Flujo B: segmentador de trazos (pendiente)"] -.-> CL
    A --> CL["Coloreado: rosado muro, azul vidrio"]
    CL -->|"color, etiquetas, hints"| S3p[("S3 procesado/")]
    CL -->|"estado y métricas"| PG
    R -.->|"si falla"| F[["Redis: fallidos"]]
    S3p --> V["Modelos de visión (D-FINE)"]
```

```mermaid
sequenceDiagram
    participant C as Cliente
    participant A as API
    participant S as S3
    participant P as Postgres
    participant R as Redis
    participant W as Worker
    C->>A: POST /v1/planos (archivo)
    A->>A: detectar formato y sha256
    A->>P: ¿ya existe ese sha256?
    alt archivo repetido
        A-->>C: 200 id existente, duplicado
    else archivo nuevo
        A->>S: guardar original
        A->>P: plano en_cola
        A->>R: XADD procesar_plano
        A-->>C: 202 id, en_cola
        R->>W: XREADGROUP
        W->>P: procesando
        W->>S: leer original
        loop cada página
            W->>W: límite y coloreado de trazos
            W->>S: color, etiquetas, hints
            W->>P: página y métricas
        end
        W->>P: listo
        W->>R: XACK
    end
    C->>A: GET /v1/planos/id
    A-->>C: estado, páginas y métricas
```

## Dentro del filtro

```mermaid
flowchart TB
    P["Página en gris, resolución completa"] --> D["Reducir a 1600 px"]
    D --> K["Tinta + líneas tenues (black-hat)"]
    K --> L["Quitar texto, marco y trazos finos; rellenar muros huecos"]
    L --> V["Cerrar vanos entre tramos alineados del muro"]
    V --> F["Relleno desde el borde con sonda: afuera"]
    F --> E["Banda del muro exterior y huella del edificio"]
    E --> S["Clasificar trazos a resolución completa: muro o vidrio"]
    P --> S
    S --> O["Salidas: color, etiquetas, hints, métricas"]
```

El límite se busca a escala reducida, pero los trazos se clasifican sobre la imagen original:
al reducir, el vidrio se aclara y desaparece (en CVC-FP pasa de gris 0 a gris 139).

## Flujo de entrenamiento

```mermaid
flowchart TB
    DS["Datasets anotados: CubiCasa5K, CVC-FP, FloorPlanCAD"] --> BD["build_dataset.py: parquet y COCO"]
    BD --> SP["Partición 30/70 por estilo de dibujo"]
    SP --> TU["ajuste_parametros: iteraciones sobre el 30 %"]
    TU --> M70["Medición en el 70 %"]
    M70 --> RP["Versión de parámetros en el registro"]
    SP --> DD["dataset_dfine: planos anotados coloreados + COCO"]
    RP -->|"activar"| DD
    DD --> DF["Afinar D-FINE con imágenes coloreadas"]
    SP --> TB["segmentador_trazos: flujo B (pendiente)"]
    TB --> MB["Medición en el 70 % por caso"]
    MB --> RB["Versión del segmentador en el registro"]
    RP -->|"activar"| WK["Workers usan la versión activa"]
    RB -.->|"activar"| WK
```

```mermaid
sequenceDiagram
    participant C as Cliente
    participant A as API
    participant P as Postgres
    participant R as Redis
    participant T as Trainer
    participant S as S3
    participant W as Worker
    C->>A: POST /v1/entrenamientos tipo ajuste_parametros
    A->>P: entrenamiento en_cola
    A->>R: XADD en la cola de entrenamiento
    A-->>C: 202 id
    R->>T: XREADGROUP
    T->>P: corriendo
    T->>T: partición 30/70, iteraciones, medición
    T->>S: modelos/parametros_filtro/version.json
    T->>P: registrar versión y métricas, entrenamiento listo
    C->>A: GET /v1/entrenamientos/id
    A-->>C: métricas y versión
    C->>A: POST /v1/modelos/id/activar
    A->>P: única versión activa del tipo
    W->>P: ¿versión activa?
    W->>S: leer parámetros nuevos
```

Reglas:

- La partición se hace una vez (`data/processed/splits.parquet`) y se reutiliza.
- Los cambios se deciden en el 30 % y se confirman en el 70 %, que nunca se usa para elegir.
- El objetivo es balanceado: promedio del F1 mediano de cada estilo, cada estilo pesa igual.
- D-FINE se entrena y se usa siempre con imágenes pasadas por la misma versión del filtro. Si
  cambia la versión activa, se vuelve a exportar el dataset y a afinar.

## Estados

```mermaid
stateDiagram-v2
    [*] --> en_cola: POST /v1/planos
    en_cola --> procesando: worker lo toma
    procesando --> listo
    procesando --> error
    listo --> [*]
    error --> [*]
```

```mermaid
stateDiagram-v2
    [*] --> en_cola: POST /v1/entrenamientos
    en_cola --> corriendo: trainer lo toma
    corriendo --> listo: versión registrada
    corriendo --> error
    listo --> activa: activar la versión
```

## Datos

```mermaid
erDiagram
    planos ||--o{ paginas : tiene
    entrenamientos ||--o| modelos : registra
    planos {
        uuid id PK
        text sha256 UK
        text nombre
        text formato
        text clave_original
        text estado
    }
    paginas {
        uuid plano_id FK
        int numero
        text estado
        jsonb salidas
        jsonb metricas
        vector embedding
    }
    entrenamientos {
        uuid id PK
        text tipo
        jsonb parametros
        text estado
        jsonb metricas
        jsonb artefactos
    }
    modelos {
        uuid id PK
        text tipo
        text version
        text uri
        jsonb metricas
        bool activo
    }
```

Claves en S3:

```
originales/<sha256>/<nombre>
procesado/<plano_id>/p<n>/{color,etiquetas,hints}.png
modelos/parametros_filtro/<version>.json
datasets/dfine/<version>/images/{train,val}/<plano>.png
datasets/dfine/<version>/annotations/{train,val}.json
```

Garantías:

- Mismo archivo, mismo plano: la API deduplica por sha256.
- Un trabajo que falla se confirma en su cola y queda en `<stream>:fallidos` con el error.
- Solo una versión activa por tipo (índice único parcial en `modelos`).
- Postgres se usa con `prepare_threshold=None` para funcionar con PgBouncer en modo transacción.
- `paginas.embedding` (pgvector) está reservado para buscar planos parecidos; aún no se llena.

## Métricas

Pruebas (`pytest --cov=filtro_planos`): 44 pruebas, todas pasan; cobertura total 72 %.

| Módulo | Cobertura | Qué cubren las pruebas |
|---|---|---|
| `envelope.py` | 97 % | Vano, esquina curva, muro a doble línea, vidriera tenue, porche con postes, plano recortado, plano oscuro |
| `ingest.py` | 96 % | PDF de varias páginas, TIFF, EXIF, transparencia, 16 bits, formato no soportado, tamaño máximo |
| `strokes.py` | 95 % | Dos clases en el límite, interior sin color, grosor del trazo intacto |
| `service/api.py` | 94 % | Ingesta, duplicados, PDF, errores 404, 415, 422 y 501, entrenamiento, activación |
| `infra/storage.py` | 92 % | Disco y S3 (simulado con moto) |
| `infra/queue.py` | 86 % | Cola en línea y Redis Streams (simulado con fakeredis), cola de fallidos |
| `training/dfine_dataset.py` | 86 % | Imágenes coloreadas y COCO |

Sin cubrir todavía: `PostgresRepository` contra una base real, el ajuste completo con datos
(`training/tuning.py`, 34 %) y la imagen de Docker.

Calidad del límite exterior, F1 mediano en el 70 % de medición (3.584 planos), sin detector:

| Estilo | Parámetros iniciales | Ajustados |
|---|---|---|
| colorful | 0,747 | 0,775 |
| CVC-FP | 0,815 | 0,865 |
| high_quality | 0,774 | 0,816 |
| architectural | 0,614 | 0,675 |
| Balanceado | 0,738 | 0,783 |

Rendimiento: 0,08 a 0,33 s por página (mediana 0,18 s) en CPU, medido en la muestra de 8 planos.

## Pendiente

- Flujo B: segmentador de trazos de dos clases (modelo base en INVESTIGACION.md).
- Medir el coloreado por clase (muro y vidrio por separado) contra las ventanas anotadas.
- Errores conocidos del flujo A: arcos de puerta junto al límite salen azules; textos y
  símbolos gruesos pegados al edificio salen rosados; vidrieras de una sola línea entre postes.
- Carga de D-FINE en `detector.py` para usos fuera de la ingesta.
