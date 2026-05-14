# Nuevas gráficas L1 y L2

Las gráficas muestran cómo la regularización "obliga" al modelo a encogerse. El modelo quiere llegar a un punto óptimo (β̂ OLS), pero hay una zona de restricción que no puede cruzar. La solución final es donde choca con esa zona.

La diferencia está en la forma de esa zona:

- **L1 (Lasso)** tiene un rombo con esquinas. La curva de error casi siempre choca con una esquina, y las esquinas caen sobre los ejes → un coeficiente queda exactamente en cero → esa variable desaparece del modelo.
- **L2 (Ridge)** tiene un círculo suave. La curva de error choca con el borde suave, nunca con un eje → todos los coeficientes se encogen, pero ninguno llega a cero → todas las variables se quedan.
