[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: Red Neuronal para el Juego Pong
## **CS2013 Programación III** · Implementación en C++

### **Descripción**

Implementación de una red neuronal multicapa en C++ para controlar el paddle en el juego Pong, utilizando aprendizaje por refuerzo. El proyecto incluye:

- Arquitectura modular de red neuronal
- Sistema de entrenamiento automático
- Paralelización con ThreadPool
- Visualización de métricas en tiempo real

---

## Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos técnicos](#requisitos-técnicos)
3. [Instalación](#instalación)
4. [Estructura del proyecto](#estructura-del-proyecto)
5. [Uso](#uso)
6. [Métricas de rendimiento](#métricas-de-rendimiento)
7. [Desarrollo](#desarrollo)
8. [Licencia](#licencia)

---

## 1. Datos generales

* **Tema**: Inteligencia Artificial para juegos clásicos
* **Autor**: Fabio Dávila Venturo
* **Grupo**: Extended_Mix
* **Repositorio**: [github.com/CS1103/projecto-final-proyecto-extended-mix](https://github.com/CS1103/projecto-final-proyecto-extended-mix)

---

## 2. Requisitos técnicos

* **Compilador**: GCC 11+ o Clang 14+ (C++20)
* **Sistema**: Linux/macOS/Windows (WSL2 recomendado para Windows)
* **Dependencias**:
  - CMake 3.18+
  - Git
* **Opcionales**:
  - Python 3.8+ (para visualización de resultados)
  - Google Colab (para entrenamiento remoto)

---

## 3. Instalación

```bash
# Clonar repositorio
git clone https://github.com/CS1103/projecto-final-proyecto-extended-mix.git
cd pong_ai
```
### 4. Estructura del proyecto:
#### 4.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ```
pong_ai/
├── include/            # Headers de la red neuronal
│   ├── utec/
│   │   ├── nn/        # Capas de red neuronal
│   │   ├── algebra/   # Operaciones tensoriales
│   │   └── parallel/  # Paralelización
├── src/
│   ├── train.cpp      # Script de entrenamiento
│   ├── data/          # Datos de entrenamiento
├── test/              # Pruebas unitarias
  ```

### 5. Uso

## Ejecución del Proyecto

### 🔍 Ejecución de Tests (CMake)

Para verificar el correcto funcionamiento de los componentes:

```bash
# Configurar el proyecto (primera vez)
mkdir build && cd build
cmake -DCMAKE_CXX_STANDARD=20 ..

# Compilar todos los tests
make

# Ejecutar todos los tests
ctest --output-on-failure
```

## 🏋️ Entrenamiento del Modelo (Compilación Directa)
Para entrenar la red neuronal con tus datos:

```bash
# Compilación optimizada
g++ -std=c++20 -O3 -Iinclude src/train.cpp -o pong_trainer -pthread

# Ejecución básica (genera output.csv)
./pong_trainer data/input.csv data/output.csv
```

### 6. Métricas de rendimiento

* **Métricas**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 1m.
  * Precisión final: 77.8%.

* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 7. Ejecución

próximamente

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
