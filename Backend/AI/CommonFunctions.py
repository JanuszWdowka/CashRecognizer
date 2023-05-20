import pandas as pd
import plotly.figure_factory as ff

def plot_confusion_matrix(cm, classes):
    """
    Funkcja do rysowania macierzy pomyłek
    :param cm: macierz pomyłek
    :param classes: Lista klas
    """
    cm = cm[::-1]
    cm = pd.DataFrame(cm, columns=classes, index=classes[::-1])

    fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), colorscale='ice',
                                      showscale=True, reversescale=True)
    fig.update_layout(width=500, height=500, title='Confusion Matrix', font_size=16)
    fig.show()