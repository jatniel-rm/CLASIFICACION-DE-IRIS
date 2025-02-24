import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
import tkinter as tk
from tkinter import simpledialog

# -------------------- Configuración global --------------------
ruta_csv = 'C:/Users/JAT/Documents/Sistemas Inteligentes/datos/archive/Iris.csv'  # Ajustar esta ruta

def cargar_datos_csv(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={
        'sepallengthcm': 'sepal_length',
        'sepalwidthcm': 'sepal_width',
        'petallengthcm': 'petal_length',
        'petalwidthcm': 'petal_width',
        'species': 'species'
    }, inplace=True)
    return df

#---------------------- Intefaz de lineal ------------#
def multi_input_dialog_lineal():
 
    dialog = tk.Toplevel()
    dialog.title("Inputs Regresión Lineal")
    dialog.geometry("350x150")
    
    species_keys = ["Iris-versicolor", "Iris-setosa", "Iris-virginica"]
    entries = {}
    
    for i, key in enumerate(species_keys):
        tk.Label(dialog, text=f"Ingrese Ancho del sépalo para {key}:", anchor="w")\
            .grid(row=i, column=0, padx=5, pady=5, sticky="w")
        entry = tk.Entry(dialog)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries[key] = entry

    result = {}
    def on_ok():
        try:
            for key in species_keys:
                result[key] = float(entries[key].get())
        except ValueError:
            for key in species_keys:
                result[key] = None
        dialog.destroy()
        
    tk.Button(dialog, text="OK", command=on_ok)\
        .grid(row=len(species_keys), column=0, columnspan=2, pady=10)
    dialog.wait_window()
    return result


def regresion_lineal():
    df = cargar_datos_csv(ruta_csv)
    
    # Obtener los tres valores en un solo cuadro
    x_usuario_dict = multi_input_dialog_lineal()
    
    # Usamos el estilo clásico de Matplotlib para evitar dependencias de Seaborn
    plt.style.use('classic')
    
    species = df['species'].unique()
    # Definimos una paleta de colores manual (puedes modificar los colores a tu gusto)
    palette = ['blue', 'green', 'red']
    
    plt.figure(figsize=(10, 6))
    
    for idx, species_name in enumerate(species):
        df_species = df[df['species'] == species_name]
        # Usamos 'petal_width' para X y 'petal_length' para Y en este ejemplo
        X_species = df_species[['petal_width']].values
        y_species = df_species['petal_length'].values
        
        modelo = LinearRegression()
        modelo.fit(X_species, y_species)
        
        # Obtener el valor ingresado para la especie actual
        x_valor = x_usuario_dict.get(species_name)
        if x_valor is None:
            # Si no se encontró, usamos el valor del primer input (por ejemplo, Iris-versicolor)
            x_valor = list(x_usuario_dict.values())[0]
        
        y_pred = modelo.predict([[x_valor]])[0]
        
        print(f'Especie: {species_name}, X = {x_valor:.2f}, Predicción Y = {y_pred:.2f}')
        
        # Graficar los datos originales usando plt.scatter
        plt.scatter(X_species.ravel(), y_species.ravel(), 
                    color=palette[idx], label=species_name, s=80)
        X_range = np.linspace(X_species.min(), X_species.max(), 100).reshape(-1, 1)
        plt.plot(X_range, modelo.predict(X_range), color=palette[idx], linewidth=2)
        plt.scatter(x_valor, y_pred, color='black', marker='X', s=120)
        plt.annotate(f"{species_name}: {y_pred:.2f}", (x_valor, y_pred),
                     textcoords="offset points", xytext=(5,5), ha="left",
                     fontsize=9, color=palette[idx])
    
    plt.xlabel('Ancho del pétalo (cm)', labelpad=10)
    plt.ylabel('Largo del pétalo (cm)', labelpad=10)
    plt.title('Regresión Lineal por Especie', pad=15)
    plt.legend(title="Especies")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def multi_input_dialog_polinomial():
    """
    Muestra un cuadro de diálogo que solicita:
      - El grado de la regresión (entero)
      - El ancho del pétalo para cada una de las tres especies:
         Iris-versicolor, Iris-setosa e Iris-virginica.
    Retorna un diccionario con las claves "grado", "Iris-versicolor", "Iris-setosa" y "Iris-virginica".
    """
    dialog = tk.Toplevel()
    dialog.title("Inputs Regresión Polinomial")
    dialog.geometry("400x200")
    
    # Definir las claves: primero el grado, luego las tres especies.
    keys = ["grado", "Iris-versicolor", "Iris-setosa", "Iris-virginica"]
    labels_text = ["Grado de la regresión:",
                   "Ancho del pétalo para Iris-versicolor:",
                   "Ancho del pétalo para Iris-setosa:",
                   "Ancho del pétalo para Iris-virginica:"]
    
    entries = {}
    for i, key in enumerate(keys):
        tk.Label(dialog, text=labels_text[i], anchor="w")\
            .grid(row=i, column=0, padx=5, pady=5, sticky="w")
        entry = tk.Entry(dialog)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries[key] = entry

    result = {}
    def on_ok():
        try:
            result["grado"] = int(entries["grado"].get())
        except ValueError:
            result["grado"] = None
        for key in keys[1:]:
            try:
                result[key] = float(entries[key].get())
            except ValueError:
                result[key] = None
        dialog.destroy()
        
    tk.Button(dialog, text="OK", command=on_ok)\
        .grid(row=len(keys), column=0, columnspan=2, pady=10)
    dialog.wait_window()
    return result

def regresion_polinomial():
    df = cargar_datos_csv(ruta_csv)
    
    # Se solicita en un único cuadro el grado y el ancho para cada especie
    inputs = multi_input_dialog_polinomial()
    grado = inputs.get("grado")
    if grado is None:
        print("El grado ingresado es inválido.")
        return

    # Estilo clásico de Matplotlib
    plt.style.use('classic')
    
    species = df['species'].unique()
    # Definir una paleta manual para tres especies (puedes ajustar los colores)
    palette = ['blue', 'green', 'red']
    
    plt.figure(figsize=(10, 6))
    
    # Se utilizará la columna 'petal_width' para X y 'petal_length' para Y
    for idx, species_name in enumerate(species):
        df_species = df[df['species'] == species_name]
        X_species = df_species[['petal_width']].values
        y_species = df_species['petal_length'].values
        
        poly = PolynomialFeatures(degree=grado)
        X_poly = poly.fit_transform(X_species)
        modelo = LinearRegression()
        modelo.fit(X_poly, y_species)
        
        # Obtener el valor ingresado para la especie actual
        x_usuario = inputs.get(species_name)
        if x_usuario is None:
            # Si el usuario no ingresó un valor válido para esa especie, se usa el de la primera especie por defecto.
            x_usuario = list(inputs.values())[1]
        
        X_usuario_poly = poly.transform([[x_usuario]])
        y_pred = modelo.predict(X_usuario_poly)[0]
        print(f'Especie: {species_name}, X = {x_usuario:.2f}, Predicción Y = {y_pred:.2f}')
        
        # Graficar los datos originales
        plt.scatter(X_species.ravel(), y_species.ravel(), 
                    color=palette[idx], label=species_name, s=80)
        # Graficar la curva de regresión
        X_range = np.linspace(X_species.min(), X_species.max(), 100).reshape(-1, 1)
        X_range_poly = poly.transform(X_range)
        plt.plot(X_range, modelo.predict(X_range_poly), color=palette[idx], linewidth=2)
        # Marcar el punto de predicción
        plt.scatter(x_usuario, y_pred, color='black', marker='X', s=120)
        plt.annotate(f"{species_name}: {y_pred:.2f}", (x_usuario, y_pred),
                     textcoords="offset points", xytext=(5,5), ha="left",
                     fontsize=9, color=palette[idx])
    
    plt.xlabel('Ancho del pétalo (cm)', labelpad=10)
    plt.ylabel('Largo del pétalo (cm)', labelpad=10)
    plt.title(f'Regresión Polinómica (Grado {grado})', pad=15)
    plt.legend(title="Especies")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------- Naive Bayes --------------------
def bayes():
    df = cargar_datos_csv(ruta_csv)
    x_usuario = simpledialog.askfloat("Input", "Ancho del petalo:")
    y_usuario = simpledialog.askfloat("Input", "Largo del petalo:")

    df["species"] = df["species"].astype("category")
    species_names = df["species"].cat.categories
    y = df["species"].cat.codes
    X = df[['petal_width', 'petal_length']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    prediccion = gnb.predict([[x_usuario, y_usuario]])[0]
    especie_predicha = species_names[prediccion]
    print(f'Clasificación: {especie_predicha}')

    # Métricas: Mostrar la matriz de confusión
    y_pred = gnb.predict(X_test)
    print("\n=== Matriz de Confusión ===")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=species_names, yticklabels=species_names)
    plt.xlabel('Predicho', labelpad=10)
    plt.ylabel('Real', labelpad=10)
    plt.title('Matriz de Confusión - Naive Bayes', pad=15)
    plt.tight_layout()
    plt.show()

    # Gráficas: frontera de decisión y dispersión de puntos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    titles = ['Entrenamiento', 'Prueba', 'Todos los datos']
    datasets = [(X_train, y_train), (X_test, y_test), (X, y)]

    # Definir la paleta personalizada para cada especie:
    custom_palette = {species_names[0]: 'green',
                      species_names[1]: 'red',
                      species_names[2]: 'blue'}

    for ax, (X_data, y_data), title in zip(axs, datasets, titles):
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
        # Convertir los códigos numéricos en nombres de especie para la gráfica:
        species_for_plot = [species_names[i] for i in y_data]
        sns.scatterplot(x=X_data[:,0], y=X_data[:,1], hue=species_for_plot,
                        palette=custom_palette, alpha=0.7, edgecolor="k", ax=ax)
        ax.scatter(x_usuario, y_usuario, color='black', marker='X', s=150,
                   label=f'Entrada: {especie_predicha}')
        ax.set_xlabel('Ancho del petalo (cm)', labelpad=10)
        ax.set_ylabel('Largo del petalo (cm)', labelpad=10)
        ax.set_title(title, pad=15)
        ax.legend()

    plt.tight_layout(pad=3.0)
    plt.suptitle('Clasificación Naive Bayes', y=1.02, fontsize=14)
    plt.show()



# -------------------- SVM --------------------
def svm():
    df = cargar_datos_csv(ruta_csv)
    x_usuario = simpledialog.askfloat("Input", "Longitud del petalo:")
    y_usuario = simpledialog.askfloat("Input", "Ancho del petalo:")

    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    species_names = le.classes_
    X = df[['petal_length', 'petal_width']].values
    y = df['species'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Transformar entrada del usuario
    input_data = scaler.transform([[x_usuario, y_usuario]])
    punto_usuario_x = input_data[0][0]
    punto_usuario_y = input_data[0][1]

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    svms = {kernel: SVC(kernel=kernel).fit(X_train, y_train) for kernel in ['linear', 'poly', 'rbf', 'sigmoid']}

    prediccion = svms['linear'].predict(input_data)[0]
    especie_predicha = species_names[prediccion]
    print(f'Clasificación: {especie_predicha}')

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    def inverse_transform(scaled, feature_idx):
        return (scaled * scaler.scale_[feature_idx]) + scaler.mean_[feature_idx]

    # Ajustar límites dinámicamente incluyendo el punto del usuario
    x_min = min(X_scaled[:, 0].min(), punto_usuario_x) - 0.5
    x_max = max(X_scaled[:, 0].max(), punto_usuario_x) + 0.5
    y_min = min(X_scaled[:, 1].min(), punto_usuario_y) - 0.5
    y_max = max(X_scaled[:, 1].max(), punto_usuario_y) + 0.5

    # Generar ticks considerando el nuevo rango
    x_ticks = np.linspace(x_min, x_max, 5)
    y_ticks = np.linspace(y_min, y_max, 5)

    for i, (kernel, model) in enumerate(svms.items()):
        plot_decision_regions(X_scaled, y, clf=model, ax=axes[i], legend=2)
        
        # Marcar punto de usuario
        axes[i].scatter(punto_usuario_x, punto_usuario_y, 
                       color='red', marker='X', s=200,
                       edgecolor='black', linewidth=1.5,
                       label=f'Entrada: {especie_predicha}')
        
        # Configurar ejes dinámicos
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(y_min, y_max)
        
        # Convertir ticks escalados a originales
        axes[i].set_xticks(x_ticks)
        axes[i].set_yticks(y_ticks)
        axes[i].set_xticklabels([f"{inverse_transform(x, 0):.1f}" for x in x_ticks], rotation=30)
        axes[i].set_yticklabels([f"{inverse_transform(y, 1):.1f}" for y in y_ticks])
        
        # boby
        axes[i].set_title(f'Kernel: {kernel.upper()}', pad=15, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Longitud del petalo (cm)', labelpad=12, fontsize=10)
        axes[i].set_ylabel('Ancho del petalo (cm)', labelpad=12, fontsize=10)
        
        # Leyendas 
        handles, labels = axes[i].get_legend_handles_labels()
        labels = [species_names[int(l)] if l.isdigit() else l for l in labels]
        axes[i].legend(handles, labels, loc='upper right', 
                      frameon=True, framealpha=0.9, 
                      edgecolor='black', title='Leyenda')

    plt.tight_layout(pad=4.5, w_pad=3.5, h_pad=3.5)
    plt.suptitle('Clasificación SVM', 
                y=0.98, fontsize=16, fontweight='bold')
    plt.show()


# -------------------- Interfaz --------------------
root = tk.Tk()
root.title("Análisis de Iris")
root.geometry("300x250")

btn_style = {'padx': 15, 'pady': 8, 'width': 20}
tk.Button(root, text="Regresión Lineal", command=regresion_lineal, padx=15, pady=8, width=20).pack(pady=5)
tk.Button(root, text="Regresión Polinomial", command=regresion_polinomial, padx=15, pady=8, width=20).pack(pady=5)
tk.Button(root, text="Naive Bayes", command=bayes, **btn_style).pack(pady=5)
tk.Button(root, text="SVM", command=svm, **btn_style).pack(pady=5)

root.mainloop()