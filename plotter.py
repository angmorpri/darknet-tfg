#!python3
# -*- coding: utf-8 -*-
"""
    Código para generar gráficas informativas sobre los resultados de un
    entrenamiento con Darknet.
    Parte de los logs que Darknet genera.
    Creado:                 08 Jun 2020
    Última modificación:    03 Nov 2020
    @author: Ángel Moreno Prieto
"""
import argparse
import matplotlib.pyplot as plt


_TRAINING_IMAGES_FILE = "./training/oxford-pet/cat-dog-train.txt"
TOTAL_IMAGES = 3688
DESCRIPTION = "Análisis de logs de entrenamiento. \nLos posibles parámetros "\
              "para los ejes son: batch, loss, avg_loss, rate, time, images y epoch"


class Batch (object):
    """Clase Batch.
    Almacena toda la información disponible por cada lote, es decir:
        * loss (float): Pérdida en entrenamiento, de la última iteración.
        * avg_loss (float): Media de pérdidas en entrenamiento.
        * learning_rate (float): Ratio de aprendizaje en la última iteración.
        * time_taken (float): Tiempo tardado en ejecutar la iteración.
        * images (float): Número de imágenes procesadas.
        * epoch (int): Número de épocas desde el inicio.
    """
    def __init__ (self, line):
        """Obtiene los parámetros a partir de la línea de log del batch."""
        params = [p.strip() for p in line.split(',')]
        self.batch = int(params[0].split(':')[0])
        self.loss = float(params[0].split(':')[1])
        self.avg_loss = float(params[1].split(' ')[0])
        self.learning_rate = float(params[2].split(' ')[0])
        self.time_taken = float(params[3].split(' ')[0])
        self.images = float(params[4].split(' ')[0])
        self.epoch = int(self.images // TOTAL_IMAGES)   # Calculada a mano.

    def getXY (self, x="batch", y="avg_loss"):
        """Devuelve dos valores indicados como X e Y por su nombre."""
        if x == "time": x = "time_taken"
        if y == "time": y = "time_taken"
        if x == "rate": x = "learning_rate"
        if y == "rate": y = "learning_rate"

        retx, rety = None, None
        try:
            retx = getattr(self, x)
            rety = getattr(self, y)
        except AttributeError:
            if retx is None:
                print(f"Error: El parámetro '{x}' para el eje X no existe")
            elif rety is None:
                print(f"Error: El parámetro '{y}' para el eje Y no existe")

        return (retx, rety)

    def __str__ (self):
        """Presenta los datos obtenidos de forma clara."""
        ret = f"Batch {self.batch}: {self.avg_loss} avg loss, {self.loss} loss,"
        ret += f" {self.learning_rate} rate, {self.time_taken} segundos,"
        ret += f" {self.images} imágenes acumuladas, {self.epoch} épocas."
        return ret


# Main
if __name__ == "__main__":
    # Argumentos en línea de comandos:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("logfile", help="archivo de log del entrenamiento")
    parser.add_argument("-x", type=str, default="batch",
                        help="parámetro del eje X")
    parser.add_argument("-y", type=str, default="avg_loss",
                        help="parámetro del eje Y")
    parser.add_argument("-xmax", type=int, default=None,
                        help="máximo valor en el eje X")
    parser.add_argument("-xmin", type=int, default=None,
                        help="mínimo valor en el eje X")
    parser.add_argument("-ymax", type=int, default=None,
                        help="máximo valor en el eje Y")
    parser.add_argument("-ymin", type=int, default=None,
                        help="mínimo valor en el eje X")
    parser.add_argument("--csv", type=str, default="results.csv",
                        help="nombre del archivo de salida CSV")
    parser.add_argument("--plot_file", type=str, default=None,
                        help="nombre del archivo de la gráfica")
    args = parser.parse_args()

    # Se recogen las líneas útiles del archivo de log:
    print("Filtrando log...")
    batches = list()
    flog = open(args.logfile, 'r')
    for line in flog:
        line = line.strip()
        if "avg" in line:
            # Línea de batch, creamos un nuevo objeto Batch
            batches.append(Batch(line))

    # Una vez se tiene la lista de batches, se preparan los valores de las
    # gráficas a imprimir.
    table = dict()      # X: (Y1, Y2, Y3, ..., Yn)
    for batch in batches:
        x, y = batch.getXY(args.x, args.y)
        if x not in table:
            table[x] = list()
        table[x].append(y)   # Un mismo valor de X puede llegar a tener varias Y

    print("Preparando CSV")
    print(f" Eje X: {args.x}, Eje Y: {args.y}")
    fcsv = open(args.csv, 'w')
    for x, ys in table.items():
        for y in ys:
            fcsv.write(f"{x}, {y}\n")
    fcsv.close()
    print(f" CSV guardado en {args.csv}")

    print("Preparando gráfica")
    xvalues = list(table.keys())
    yvalues = list()
    for ys in table.values():
        noninf = [y for y in ys if y == y]  # Elimina NaNs
        if noninf:
            yvalues.append(noninf[-1])
    xmax = args.xmax or float("inf")
    xmin = args.xmin or float("-inf")
    ymax = args.ymax or float("inf")
    ymin = args.ymin or float("-inf")
    subxvalues = list()
    subyvalues = list()
    for x, y in zip(xvalues, yvalues):
        if (xmin < x < xmax) and (ymin < y < ymax):
            subxvalues.append(x)
            subyvalues.append(y)
    fig = plt.figure()
    plt.plot(subxvalues, subyvalues)
    plt.grid(True, linestyle=':')
    plt.xlabel(f"{args.x.replace('_', ' ').capitalize()}")
    plt.ylabel(f"{args.y.replace('_', ' ').capitalize()}")
    plt.tight_layout()
    if not args.plot_file:
        args.plot_file = f"plot-{args.x}-{args.y}.png"
    fig.savefig(args.plot_file, dpi=1000)
    print(f" Gráfica guardada en {args.plot_file}")
    print("Finalizado!")
