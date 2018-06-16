import jensen_shannon
import seaborn
import numpy as np
import pandas
from math import pi
from bokeh.io import output_file
from bokeh.palettes import BuPu
from matplotlib import cm
import matplotlib as mpl
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
)



def make_heatmap(data, title, outfile):
    """
    Create a heatmap for a distance DataFrame

    @args:
    - the DataFrame
    - a title for the plot
    - a path for the output file *.html

    @output:
    - the heatmap is stored in the outfile
    """
    output_file(outfile, title=title)
    l1 = list(data.columns)
    l2 = list(data.index)

    # reshaping
    df = pandas.DataFrame(data.stack(), columns=['dist']).reset_index()
    df = df.fillna(0)

    # color setting
    colormap =cm.get_cmap("BuPu")
    colors = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    mapper = LinearColorMapper(palette=colors, low=df.dist.min(), high=df.dist.max())

    source = ColumnDataSource(df)
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    p = figure(title=title,
               x_range=l1, y_range=sorted(l2, reverse=True),
               # x_axis_location="above", plot_width=2000, plot_height=1700, # lang
               # x_axis_location="above", plot_width=3500, plot_height=3000, # treebanks
               # x_axis_location="above", plot_width=1200, plot_height=900, # treebanks romanes
               x_axis_location="above", plot_width=800, plot_height=600, # langs romanes

               tools=TOOLS, toolbar_location='below')
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "15pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3
    p.rect(x="level_1", y="level_0", width=1, height=1,
           source=source,
           fill_color={'field': 'dist', 'transform': mapper},
           line_color=None)
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="15pt",
                         ticker=BasicTicker(desired_num_ticks=10),
                         label_standoff=8, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    p.select_one(HoverTool).tooltips = [
         ('l1', '@level_0'),
         ('l2', '@level_1'),
         ('distance', '@dist'),
    ]

    # automatically opens
    show(p)
    print("done..")

if "__main__" == __name__:
    # for JSD unipos
    # data = jensen_shannon.JSD_pos_dataframe("../memoire_outfiles/distrib_pos.csv")
    # make_heatmap(data, 'JSD based on unipos distribution', "../memoire_outfiles/plots/JSD_unipos.html")

    # for euclidean distance dep direction
    # data = pandas.read_csv("../memoire_outfiles/euclidean_distances_dependency_direction.csv", sep="\t", index_col=0)
    # make_heatmap(data, 'Euclidean distance based on dep direction vectors', "../memoire_outfiles/plots/euclidean_distances_dependency_direction.html")


    # for euclidean distance dep length
    data = pandas.read_csv("../memoire_outfiles/trigrammes-pos/jsd_distance_3pos_bylang_rom.csv", sep="\t", index_col=0)
    make_heatmap(data, 'Distance de Jensen-Shannon entre les distributions de trigrammes de catégories morpho-syntaxiques des langues romanes', "../memoire_outfiles/plots/trigrammes-pos/jsd_distance_3pos_bylang_rom.html")
