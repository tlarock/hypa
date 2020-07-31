from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
plt.rcParams['font.sans-serif'] =  ['Roboto Condensed', 'Inter UI']+ \
                                    plt.rcParams['font.sans-serif'] 


def set_style():
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (4.8, 3.6)
    # plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0
    plt.rcParams['savefig.dpi'] = 300
    #plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 
    #    'Inter UI', 'Roboto', 'Noto Sans','Helvetica', 'Arial'] + \
    #    plt.rcParams['font.sans-serif'] 
    plt.rcParams['font.size'] = 12

def sequential_color_list():
    return ["#8cd3ff",
            "#f2d249",
            "#d998cb",
            "#b6d957",
            "#5cbae6",
            "#fac364",
            "#93b9c6",
            "#ccc5a8",
            "#52bacc",
            "#dbdb46",
            "#98aafb",
            "#5cbae6"]



cdict_overunder = {
     'red':   (
               (0.0, 1.00, 1.00),
               (0.1, 1.00, 1.00),
               (0.9, 1.00, 0.60),
               (1.0, 0.60, 0.60)
               ),

     'green': (
               (0.0, 0.28, 0.28),
               (0.1, 0.28, 1.00),
               (0.9, 1.00, 0.95),
               (1.0, 0.95, 0.95)
               ),

     'blue':  (
               (0.00, 0.23, 0.23),
               (0.10, 0.23, 1.00),
               (0.90, 1.00, 0.23),
               (1.00, 0.23, 0.23)
               ),

     'alpha': (
               (0.00, 1.0, 1.0),
               (0.05, 0.3, 0.3),
               (0.10, 0.0, 0.0),
               (0.90, 0.0, 0.0),
               (0.95, 0.3, 0.3),
               (1.00, 1.0, 1.0)
               ),
    }

cm_over_under = LinearSegmentedColormap('OverUnder', cdict_overunder)
plt.register_cmap(cmap=cm_over_under)

set_style()
