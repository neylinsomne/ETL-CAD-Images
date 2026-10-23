# Investigación: repositorios y modelo base para el flujo B

Objetivo del flujo B: un modelo que clasifique cada trazo del límite exterior en dos clases,
muro (rosado) y vidrio en el límite (azul: ventanas, muros cortina, vidrieras), sobre planos
raster. Restricciones: GPU RTX 5060 de 8 GB (Blackwell, necesita PyTorch 2.7 o superior con
CUDA 12.8) y licencias que no bloqueen un uso comercial futuro.

Revisión hecha en octubre de 2026 sobre las páginas, la API de GitHub y las fichas de cada
proyecto. Lo marcado "sin verificar" no se pudo confirmar en una fuente primaria.

## Lo que dicen nuestros datos

Medido en 120 planos anotados (CubiCasa5K y CVC-FP):

| | Muro exterior | Ventana exterior |
|---|---|---|
| Gris típico del trazo | 0 a 105 | 122 a 160 |
| Parte del trazo que solo aparece con detección de líneas tenues | 0 a 19 % | 24 a 49 % |

En CVC-FP la ventana es negra a resolución completa y queda en gris 139 al reducir a 1600 px.
Conclusión: el vidrio se pierde si se reduce la imagen o si no se buscan líneas tenues.

## Mapa de clases de FloorPlanCAD (nuestra copia, numeración v2)

Fuente: `RemapDict` de CADTransformer (`config/anno_config.py`) aplicado a la lista v1, y
comprobado sobre 400 archivos nuestros (los ids sin instancia son solo 1, 2, 33, 34 y 35).

| Ids v2 | Clase |
|---|---|
| 1 | muro |
| 2 | muro cortina |
| 3 a 8 | puertas (simple, doble, corredera, plegable, giratoria, enrollable) |
| 9 a 12 | ventana, ventana saliente, ventana ciega, símbolo de vano |
| 13 a 29 | mobiliario y artefactos |
| 30 a 32 | escalera, ascensor, escalera mecánica |
| 33 | baranda |
| 34 | filas de sillas |
| 35 | estacionamiento |

Todos los modelos publicados usan la numeración v1 (muro 33, muro cortina 34). Para nuestras
dos clases: rosado = 1; azul = 2, 9, 10. Dudosos: 11, 12, 5 (corredera, suele ser vidriada) y
33. FloorPlanCAD no marca qué es exterior: hay que derivarlo (relleno desde el borde).

## Repositorios sobre imágenes (raster)

| Repositorio | Qué aporta | Licencia | Corre en la 5060 | Pesos | Utilidad |
|---|---|---|---|---|---|
| CubiCasa/CubiCasa5k | Modelo multitarea con clases muro y ventana; anotación en SVG | CC BY-NC (código, datos y pesos) | Con port medio (PyTorch 1.0) | Sí | 4/5: fuente de etiquetas |
| aliasstudio/mitunet | U-Net con atención scSE y pérdida Tversky para muros | Código MIT; pesos CC BY-NC; encoder MiT de NVIDIA no comercial | Sí (torch 2.8) | Sí, 257 MB | 4/5: receta de entrenamiento |
| Chunling1/OpenBIM-FloorPlan-AI | U-Net ResNet34 con muro, ventana y puerta; aleatorización de dominio | Código MIT; pesos no comerciales (CubiCasa) | Sí (PyTorch 2) | Sí | 4/5: línea base rápida |
| MLSTRUCT/MLStructFP | U-Net de muros en planos de 6500 a 9500 px | Código MIT; datos por formulario | Port alto (TF 2.2) | Sí | 2/5: idea de mosaicos a resolución completa |
| zlzeng/DeepFloorplan y ports | Límite de ambientes (muro, puerta y ventana) | GPL (TF); port PyTorch MIT sin pesos | Port alto | Parcial | 2/5 |
| art-programmer/FloorplanTransformation | Raster a vector por esquinas | MIT | Port alto (Torch7) | Sí | 1/5: sin clase ventana |
| Hugging Face: hallelu, phungpx | U-Net chica sin pesos; muros sobre RMBG-1.4 | MIT con datos CubiCasa; BRIA no comercial | Sí | Parcial | 1 a 2/5 |

Dato útil: el dataset Floor Plan CIS de MitUNet (500 imágenes, Zenodo 17871079) es CC BY 4.0,
o sea que sí permite uso comercial.

## Repositorios vectoriales (cada trazo con su clase)

| Repositorio | Qué aporta | Licencia | Corre en la 5060 | Pesos | Utilidad |
|---|---|---|---|---|---|
| VITA-Group/CADTransformer | CNN sobre la imagen y un transformer sobre los trazos, con rasgos tomados de la imagen en cada trazo | MIT | Port fácil (sin extensiones CUDA) | No | 4/5: arquitectura de la idea C |
| WesKwong/VecFormer (NeurIPS 2025) | Tokens de línea; el mejor en muros y muros cortina (stuff PQ 85,9) | Apache-2.0 | Difícil (flash-attn, spconv) | No | 3/5 |
| ArchiAI-LAB/ArchCAD (DPSS) | Fusión con compuerta entre rasgos de imagen y de trazo; dataset con clase vidrio | Solo uso académico; datos CC BY-NC con aprobación | Con parches (otro usuario con 5060 Ti y WSL2) | No | 3/5: idea de fusión |
| DrZiji/VecFloorSeg | Dualidad línea y región: un trazo exterior toca la región de afuera | Sin licencia | Port medio a alto (PyG) | Sin verificar | 3/5: idea para etiquetar exterior |
| nicehuster/SymPoint y SymPointV2 | Cada trazo como punto; muy bueno en objetos, débil en muros (stuff PQ 48) | No comercial / sin licencia | Difícil (extensión CUDA vieja) | Sí | 2/5 |
| GAT-CADNet | Grafo de trazos con atención | Sin código oficial | No aplica | No | 2/5 |

No existe un repositorio público que haga esto sobre imágenes raster con clases muro y vidrio.

## Piezas de construcción

| Pieza | Para qué | Licencia | Nota |
|---|---|---|---|
| segmentation_models_pytorch | U-Net y pérdidas (Tversky, Focal, Dice) | MIT, salvo el encoder MiT (NVIDIA, no comercial) | Base recomendada |
| timm | Encoders ResNet34 y EfficientNet-B3 preentrenados | Apache-2.0 en código y en esas fichas | ConvNeXt-V2 es CC BY-NC: evitar |
| SegFormer (NVIDIA) | Encoder MiT | No comercial | Evitar |
| MIC-DKFZ/Skeleton-Recall | Pérdida que mantiene continuas las líneas finas | Apache-2.0 | Se porta en unas 50 líneas; esqueleto por clase |
| jocpae/clDice | Pérdida de continuidad alternativa | MIT | Más cara y floja en varias clases |
| MIC-DKFZ/nnUNet v2 | U-Net autoconfigurada como línea base | Apache-2.0 | 8 GB queda justo |
| OpenCV LSD y FastLineDetector | Vectorizar trazos | Apache-2.0 | Duplican cada línea fina (detectan bordes) |
| cvg/DeepLSD | Detección de segmentos aprendida | MIT | Sin instalación documentada en Windows |
| Peterande/D-FINE | Detector para después de la ingesta | Apache-2.0 | Usar pesos solo COCO (los de Objects365 tienen otros términos); entrada 640 por defecto |
| obss/SAHI | Inferencia por mosaicos | MIT | No soporta D-FINE tal cual: hay que parchear el puntaje |

## Mejores ideas

1. **Fábrica de etiquetas por trazo.** CubiCasa5K (muro externo y sus ventanas), CVC-FP
   (`outerP`) y FloorPlanCAD v2 (1 rosado; 2, 9 y 10 azul) se convierten en máscaras de
   píxeles de trazo. Lo exterior se decide con el relleno desde el borde que ya tenemos.
2. **Pérdida solo sobre trazos.** Lo que se colorea son trazos, así que la pérdida se calcula
   sobre los píxeles de tinta (más una muestra de fondo), no sobre áreas.
3. **Tversky más Skeleton Recall.** Tversky con alfa 0,3 y beta 0,7 favorece no perder vidrio;
   Skeleton Recall, por clase, evita que la línea fina se corte.
4. **Mosaicos a resolución completa.** Nunca reducir el plano antes del modelo.
5. **Aleatorización de dominio con tonos.** Variar gris, grosor, ruido y escaneo; en especial
   aclarar líneas de vidrio, que es justo lo que falla.
6. **El flujo A como pista y respaldo.** Su zona del límite puede entrar como canal extra al
   modelo, y sigue siendo el resultado cuando el modelo no está seguro.
7. **Más adelante (idea C):** vectorizar el raster y clasificar cada trazo con rasgos de imagen
   tomados sobre él (CADTransformer) y fusión con compuerta (DPSS).

## Recomendación de modelo base

U-Net de `segmentation_models_pytorch` con encoder `tu-resnet34.a1_in1k` (alternativa:
`tu-efficientnet_b3.ra2_in1k`), tres clases (otro, muro, vidrio), entrada en gris más el canal
opcional de la zona del flujo A, mosaicos de 512 a 1024 px a resolución completa.

- ResNet y EfficientNet conservan los rasgos a la mitad de resolución, que son los que ven
  una línea de 1 px; MiT y ConvNeXt empiezan a un cuarto.
- Código MIT y Apache, pesos Apache. Advertencia general de timm: los términos de ImageNet son
  no comerciales y su efecto sobre pesos preentrenados no está claro. Si legal lo rechaza, se
  entrena desde cero: un dibujo en blanco y negro necesita poco de ImageNet.
- Pérdida: entropía cruzada con pesos por clase, Tversky (0,3 y 0,7) y Skeleton Recall por clase.
- Medición: la misma partición 30/70 del filtro, por clase (muro y vidrio por separado) y por
  tipo de caso, comparado contra el flujo A.

Los datos anotados disponibles son todos no comerciales: el modelo resultante sirve para
investigar y medir. Para producción, reentrenar con planos propios etiquetados.

## Para D-FINE, después de la ingesta

- Pesos COCO, configuración `custom_detection.yml`; en Windows, `python train.py` sin torchrun.
- Entrada de 640 por defecto: el vidrio coloreado de 1 px se vuelve a aclarar al reducir. Subir
  la resolución (1024 o más) o trabajar por mosaicos. Otra opción es engrosar solo un poco el
  trazo coloreado en la copia que entra a D-FINE (no en la salida del filtro).
