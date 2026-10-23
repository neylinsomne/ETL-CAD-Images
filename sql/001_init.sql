-- Esquema de metadatos del filtro de planos (Postgres 15+, detrás de PgBouncer).
-- Los archivos (originales y salidas) viven en S3; aquí solo claves y métricas.

CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector: reservado para buscar planos parecidos

CREATE TABLE IF NOT EXISTS planos (
    id              uuid PRIMARY KEY,
    sha256          text NOT NULL UNIQUE,          -- mismo archivo dos veces = mismo plano
    nombre          text NOT NULL,
    formato         text NOT NULL,                 -- pdf | png | jpeg | tiff | bmp | webp
    bytes           bigint NOT NULL,
    clave_original  text NOT NULL,                 -- clave en S3
    estado          text NOT NULL,                 -- en_cola | procesando | listo | error
    error           text,
    creado          timestamptz NOT NULL DEFAULT now(),
    actualizado     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paginas (
    plano_id    uuid NOT NULL REFERENCES planos (id) ON DELETE CASCADE,
    numero      int NOT NULL,
    estado      text NOT NULL,                     -- ok | cropped | no_footprint | unbounded | too_dark
    salidas     jsonb NOT NULL,                    -- {"color": clave S3, "etiquetas": ..., "hints": ...}
    metricas    jsonb NOT NULL,                    -- grosor, píxeles de muro y vidrio, huella (WKT)...
    embedding   vector(512),                       -- reservado: huella visual de la página
    creado      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (plano_id, numero)
);

CREATE INDEX IF NOT EXISTS paginas_estado_idx ON paginas (estado);

-- Trabajos de entrenamiento lanzados por la API (POST /v1/entrenamientos).
CREATE TABLE IF NOT EXISTS entrenamientos (
    id          uuid PRIMARY KEY,
    tipo        text NOT NULL,                     -- ajuste_parametros | dataset_dfine | segmentador_trazos
    parametros  jsonb NOT NULL,
    estado      text NOT NULL,                     -- en_cola | corriendo | listo | error
    metricas    jsonb,
    artefactos  jsonb,                             -- claves S3 y modelo registrado
    error       text,
    creado      timestamptz NOT NULL DEFAULT now(),
    actualizado timestamptz NOT NULL DEFAULT now()
);

-- Versiones de parámetros y modelos que usan los workers (flujo de entrenamiento).
CREATE TABLE IF NOT EXISTS modelos (
    id          uuid PRIMARY KEY,
    tipo        text NOT NULL,                     -- parametros_filtro | dataset_dfine | segmentador_trazos | dfine
    version     text NOT NULL,
    uri         text NOT NULL,                     -- clave S3 del archivo de parámetros o pesos
    metricas    jsonb NOT NULL,                    -- F1 balanceado en el 70 %, por estilo...
    activo      boolean NOT NULL DEFAULT false,
    creado      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tipo, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS modelos_un_activo_por_tipo ON modelos (tipo) WHERE activo;
