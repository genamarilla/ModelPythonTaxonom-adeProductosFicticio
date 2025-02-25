# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:26:27 2025

@author: Gene
"""

# -*- coding: utf-8 -*-
"""
Red de Taxonomía de Productos. Modelo Ficticio
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Cargar datos desde Excel
file_path = r"C:\\Users\\Gene\\Documents\\TFG 2024\\BDD_ficticia.xlsx"
EXP_df = pd.read_excel(file_path, header=0, index_col=0)

# Extraer países y productos
country_codes = EXP_df.index
product_codes = EXP_df.columns
EXP = EXP_df.values

# Cálculo de VCR y Mcp
Vc = np.sum(EXP, axis=1)  # Diversidad de cada país (dc)
Vp = np.sum(EXP, axis=0)  # Ubicuidad de cada producto (up)
SumTotal = np.sum(Vc)
VCR = (EXP / Vc[:, None]) / (Vp / SumTotal)
Mcp = (VCR >= 1).astype(int)

# Cálculo de la matriz Bpp' (antes Mpp')
ubiquity = np.sum(Mcp, axis=0)  # up
diversity = np.sum(Mcp, axis=1)  # dc

Bpp_prime = np.zeros((Mcp.shape[1], Mcp.shape[1]))  # Inicializar matriz

for p in range(Mcp.shape[1]):
    for p_prime in range(Mcp.shape[1]):
        if p != p_prime:  # Evitar calcular auto-conexiones
            # Numerador: suma sobre países, dividiendo por diversidad (dc)
            numerator = np.sum(Mcp[:, p] * Mcp[:, p_prime] / diversity)
            # Denominador: máximo entre las ubicuidades de los productos (up)
            denominator = max(ubiquity[p], ubiquity[p_prime])
            if denominator != 0:  # Evitar divisiones por cero
                Bpp_prime[p, p_prime] = numerator / denominator

# Redondear los valores de Bpp' para comparar con los resultados esperados
Bpp_prime = np.round(Bpp_prime, decimals=10)

# Mostrar la matriz para validar
print("Matriz Bpp':")
print(pd.DataFrame(Bpp_prime, index=product_codes, columns=product_codes))

# Crear la red de taxonomía de productos dirigida y jerárquica eliminando redundancias
def plot_taxonomy_hierarchy(Bpp, product_codes, final_complexity):
    G = nx.DiGraph()

    # Agregar nodos con sus complejidades
    for i in range(len(product_codes)):
        if final_complexity[i] > 0:
            G.add_node(product_codes[i], size=final_complexity[i])

    # Agregar aristas basadas en la matriz Bpp (del producto menos complejo al más complejo)
    for i in range(len(product_codes)):
        for j in range(len(product_codes)):
            if Bpp[i, j] > 0.1 and final_complexity[i] < final_complexity[j]:  # Umbral para considerar una conexión
                G.add_edge(product_codes[i], product_codes[j], weight=Bpp[i, j])

    # Eliminar conexiones redundantes según la teoría de Tacchella
    for node in G.nodes:
        predecessors = list(G.predecessors(node))
        for p1 in predecessors:
            for p2 in predecessors:
                if p1 != p2 and G.has_edge(p1, p2):
                    G.remove_edge(p1, node)  # Eliminar redundancia

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("El grafo no tiene nodos o aristas suficientes para visualizar.")
        return

    print(f"Grafo generado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

    # Calcular la jerarquía utilizando un layout jerárquico
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")

    plt.figure(figsize=(12, 12))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=[G.nodes[n]['size'] * 1000 for n in G.nodes],
        font_size=8,
        edge_color="gray",
        arrowsize=20  # Tamaño de las flechas
    )
    plt.title("Red Dirigida de Taxonomía de Productos - Modelo Ficticio")
    plt.show()

# Cálculo de Fitness y Complejidad
N = 1000  # Número de iteraciones
FcN = np.ones((Mcp.shape[0], N))
QpN = np.ones((Mcp.shape[1], N))
for k in range(1, N):
    FcN[:, k] = np.sum(Mcp * QpN[:, k - 1], axis=1)
    FcN[:, k] /= np.mean(FcN[:, k])
    QpN[:, k] = 1 / np.sum(Mcp * (1 / FcN[:, k - 1])[:, None], axis=0)
    QpN[:, k] /= np.mean(QpN[:, k])

final_fitness = FcN[:, -1]
final_complexity = QpN[:, -1]

plot_taxonomy_hierarchy(Bpp_prime, product_codes, final_complexity)

# Guardar resultados en un archivo Excel
output_file = "Resultados_Analisis_Economico_ModeloFicticio.xlsx"
with pd.ExcelWriter(output_file) as writer:
    pd.DataFrame(VCR, index=country_codes, columns=product_codes).to_excel(writer, sheet_name="VCR")
    pd.DataFrame(Mcp, index=country_codes, columns=product_codes).to_excel(writer, sheet_name="Mcp")
    pd.DataFrame(Bpp_prime, index=product_codes, columns=product_codes).to_excel(writer, sheet_name="Bpp'")
    pd.DataFrame({"country_id": country_codes, "fitness": final_fitness}).to_excel(writer, sheet_name="Fitness", index=False)
    pd.DataFrame({"product_id": product_codes, "complexity": final_complexity}).to_excel(writer, sheet_name="Complexity", index=False)

print(f"Resultados guardados en {output_file}")
