Last Update: 2023-11-17 8:35PM

Para esta mejora se utilizo el siguinete dataset:

https://universe.roboflow.com/train-mmixf/caparoja

Este data set contiene alrededor de 43 imagenes en color rojo sin augmentation.


Paso a paso para generar nuestro modelo utilizando un dataset de roboflow:

Paso 0:
- Ingresar al archivo requirements.txt para ver las instalaciones necesarios en los proximos pasos.

Paso 1:
- Ingresamos a la direccion https://universe.roboflow.com/train-mmixf/caparoja y hacemos click
en el boton "Download this dataset"
- Ingresamos al sitio web https://roboflow.com y creamos una cuenta o iniciamos sesion.
- Creamos un nuevo espacio de trabajo o proyecto y seleccionamos el tipo como Instance Segmentation,
ingresamos la description y nombre del proyecto.

Paso 2:
- Descomprimimos el archivo zip descargado en el paso 1.
- En el dashboard del proyecto creado en el paso 1 arrastramos TODOS los archivos generados al descomprimir
el archivo zip, esto nos permitira visualizar todas las imagenes.
- Damos click en "Saven and continue" para cargar los archivos.
- Procedemos a configurar la version actual del dataset. Es recomendable crear data augmentation para asi tener un
mejor entrenamiento del modelo.
- Damos click en general.

Paso 3: 
- Una vez se genere la version del proyecto, damos click en Export Dataset.
- Seleccionamos la opcion YOLOV8 y "Show Code". Eso con el fin de poder conectarnos directamente a roboflow cuando 
estemos en tiempo de ejecucion. NOTA: Se puede descargar el .zip lo cual seria un uso igual.
PRECAUCION: El codigo generado contiene el token unico de proyecto que da acceso al proyecto en general a cualquier persona.

Paso 4:
- Pegamos el codigo generado en el paso anterior en google collab o en nuestro archivo python.
- Antes de ejecutar nuestro codigo hacemos la siguiente instalacion requerida: !pip install roboflow roboflow se usara para conectarse
al repositorio de nuestro entorno de trabajo o proyecto.
- Una vez acabe la descarga de nuestro proyecto, se habran descargado varias carpetas las cuales incluyen: una carpeta train y otra test con las imagenes y labels respecitvos,
el archivo data.yaml que permitira ejecutar nuestro modelo y dos archivos de texto con instrucciones.

Paso 5:
- Una vez se haya terminado de descargar el proyecto, se ejecuta por unica vez el comando: !pip install ultralytics para instalar los paquetes necesarios para usar YOLO.
- NOTA: Antes de entrenar el modelo es recomendable abrir el archivo data.yaml para verificar que las rutas de test, train y val correspondan a rutas validas.
- Ahora es el momento de entrenar nuestro modelo usando el siguiente comando: !yolo task=segment mode=train epochs=160 data={PATH}/data.yaml model=yolov8m-seg.pt imgsz=640 batch=2 . El comando 
anterior entrenera nuestro modelo para segmentar las predicciones y crear mascaras de los objetos encontrados. El numero de epocas acordado es de 160 por motivos de accuracy encontrado para imagenes de tamano 
de 640 (predispuestas en el dataset seleccionado).
- Despues del entrenamiento del modelo, se generara un archivo de nombre best.pt . La ruta de este archivo la arrojara el modelo al terminar su entrenamiento. Generalmente 
tiene esta ruta: runs/segment/train/weights/best.pt
- El modelo best.pt sera el que se usara a lo largo del proyecto.