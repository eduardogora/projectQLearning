# README.md

## Juego de Laberinto con Q-Learning

Este proyecto es un juego de laberinto implementado en Python utilizando la biblioteca `pygame`. El objetivo del juego es que un agente (representado por un bloque) recorra el laberinto y coloree todas las celdas de camino (`ROAD`) sin chocar con los muros (`WALL`). El agente utiliza el algoritmo de **Q-learning** para aprender a moverse de manera óptima en el laberinto.

---

## Requisitos

Para ejecutar este proyecto, necesitas tener instalado lo siguiente:

- **Python 3.x**: Asegúrate de tener Python instalado en tu sistema.
- **Pygame**: La biblioteca `pygame` se utiliza para la interfaz gráfica del juego.
- **NumPy**: Se utiliza para manejar matrices y operaciones numéricas.
- **Matplotlib**: Se utiliza para graficar el historial de éxito del agente.

Puedes instalar las dependencias necesarias ejecutando el siguiente comando:

```bash
pip install pygame numpy matplotlib
```

---

## Estructura del Proyecto

El proyecto consta de los siguientes archivos:

1. **`game.py`**: El archivo principal que contiene la lógica del juego y el algoritmo de Q-learning.
2. **`easyMazes.py`**: Contiene los laberintos fáciles y las posiciones iniciales del agente.
3. **`mediumMazes.py`**: Contiene los laberintos de dificultad media y las posiciones iniciales del agente.
4. **`gameconfig.json`**: Archivo de configuración que define parámetros como el número máximo de pasos, episodios, y si se deben cargar pesos preentrenados.
5. **`Qlearning.json`**: Archivo de configuración que define los parámetros del algoritmo Q-learning, como la tasa de aprendizaje, el factor de descuento y la exploración (epsilon).

---

## Configuración

### Archivo `gameconfig.json`

Este archivo contiene los siguientes parámetros:

- **`fps`**: Los fotogramas por segundo del juego.
- **`max_steps`**: El número máximo de pasos permitidos por episodio.
- **`max_episodes`**: El número máximo de episodios de entrenamiento.
- **`size`**: El tamaño de la Q-table (depende del tamaño del laberinto).
- **`load_weights`**: Si es `true`, el juego cargará una Q-table previamente entrenada desde `Checkpoints/Q_table.npy`. Si es `false`, comenzará con una Q-table vacía.
- **`enable_render`**: Si es `true`, se habilitará la renderización gráfica del juego. Si es `false`, el juego se ejecutará en modo "headless" (sin interfaz gráfica).

### Archivo `Qlearning.json`

Este archivo contiene los siguientes parámetros del algoritmo Q-learning:

- **`max_epsilon`**: El valor máximo de epsilon (exploración).
- **`min_epsilon`**: El valor mínimo de epsilon.
- **`epsilon_decay`**: La tasa de decaimiento de epsilon.
- **`learning_rate`**: La tasa de aprendizaje del algoritmo.
- **`discount_factor`**: El factor de descuento para las recompensas futuras.
- **`epsilon`**: El valor inicial de epsilon.

---

## Ejecución del Juego

Para ejecutar el juego, sigue estos pasos:

1. **Clona el repositorio** o descarga los archivos del proyecto.
2. **Navega al directorio** donde se encuentran los archivos.
3. **Ejecuta el archivo `game.py`**:

```bash
python game.py
```

### Modo de Entrenamiento

- Si `load_weights` es `false` en `gameconfig.json`, el juego comenzará con una Q-table vacía y entrenará al agente desde cero.
- Al final del entrenamiento, la Q-table se guardará en `Checkpoints/Q_table.npy`.

### Modo de Prueba

- Si `load_weights` es `true`, el juego cargará la Q-table previamente entrenada desde `Checkpoints/Q_table.npy` y la utilizará para tomar decisiones.

---

## Resultados

Al finalizar el entrenamiento, el juego mostrará:

1. **Gráfico de éxito**: Un gráfico que muestra el número de pasos necesarios para completar cada episodio.
2. **Q-table**: La tabla Q final se imprimirá en la consola.
3. **Número de victorias**: El número de veces que el agente completó el laberinto correctamente.

---

## Personalización

### Laberintos

Puedes agregar o modificar laberintos en los archivos `easyMazes.py` y `mediumMazes.py`. Cada laberinto es una matriz donde:

- `0` representa un muro (`WALL`).
- `2` representa un camino (`ROAD`).
- `1` representa la posición inicial del agente (`PLAYER`).

### Parámetros de Q-learning

Puedes ajustar los parámetros del algoritmo Q-learning en el archivo `Qlearning.json` para mejorar el rendimiento del agente.

---

## Ejemplo de Laberinto

Aquí hay un ejemplo de un laberinto en `easyMazes.py`:

```python
grids = [
    np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ]),
]
```

---

## Contribuciones

Si deseas contribuir a este proyecto, ¡eres bienvenido! Puedes:

- Agregar nuevos laberintos.
- Mejorar el algoritmo de Q-learning.
- Optimizar el código o agregar nuevas características.

---

## Licencia

Este proyecto está bajo la licencia MIT. Siéntete libre de usarlo y modificarlo según tus necesidades.

---

¡Gracias por usar este proyecto! Si tienes alguna pregunta o sugerencia, no dudes en contactarme. 😊

---

