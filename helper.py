import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
sns.set()


def calculate_weighted_rating(df, C, m):
    v = df['vote_count']
    R = df['vote_average']
    return ((v/(v+m))*R) + ((m/(v+m))*C)

def plot_graph(attrs):

    fig = plt.figure(figsize=(16,9))

    fig.suptitle(attrs["super_title"], fontsize=16)

    # Plot graph for Order base on IMDB's weighted rating
    ax1 = plt.subplot(1,2,1)

    ax1_bar = ax1.barh(attrs["ax1_movie_title"], attrs["ax1_data"], color=attrs["ax1_color"])
            
    rects1 = ax1.patches

    # Place a label for each bar
    for rect in rects1:

        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        label = attrs["ax1_annotate_label"].format(x_value)

        plt.annotate(
            label,
            (x_value, y_value),
            xytext=(2, 0),
            textcoords='offset points',
            va='center',
            ha='left',
            color = attrs["ax1_color"])
        
    ax1.set_title(attrs["ax1_title"])

    ax1.set_xlabel(attrs["ax1_xlabel"])

    ax1.invert_yaxis()

    if attrs["ax1_set_scale"]:

        ax1.set_xscale('log')

    ax1.xaxis.set_minor_formatter(StrMethodFormatter(attrs["ax1_xaxis_formatter"]))

    plt.xlim(attrs["ax1_xlim"])

    # Plot graph for Order base on popularity column

    ax2 = plt.subplot(1,2,2)

    bar2 = ax2.barh(attrs["ax2_movie_title"], attrs["ax2_data"], color=attrs["ax2_color"])

    rects2 = ax2.patches

    # Place a label for each bar
    for rect in rects2:

        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        label = attrs["ax2_annotate_label"].format(x_value)

        plt.annotate(
            label,
            (x_value, y_value),
            xytext=(2, 0),
            textcoords='offset points',
            va='center',
            ha='left',
            color = attrs["ax2_color"])

    ax2.set_title(attrs["ax2_title"])

    ax2.set_xlabel(attrs["ax2_xlabel"])

    ax2.invert_yaxis()

    if attrs["ax2_set_scale"]:

        ax2.set_xscale('log')

    ax2.xaxis.set_minor_formatter(StrMethodFormatter(attrs["ax2_xaxis_formatter"]))

    plt.xlim(attrs["ax2_xlim"])

    plt.tight_layout(pad=2)

    plt.show()


def get_top_tier_recommendation_from_title(movie_df, title, simility_matrix, num_of_movie):

    # a series with title as key, index as values
    indices = pd.Series(movie_df.index, index=movie_df['title']).drop_duplicates()

    movie_index = indices[title]

    movie_list = list(enumerate(simility_matrix[movie_index]))

    sorted_movie_list = sorted(movie_list, key=lambda x: x[1], reverse=True)

    # Pick top tier but exclude itself
    top_tier_list = sorted_movie_list[1:(num_of_movie + 1)]

    top_tier_index_list = []
    simility_value_list = []

    for i in range(num_of_movie):
        top_tier_index_list.append(top_tier_list[i][0])
        simility_value_list.append(top_tier_list[i][1])

    top_tier_df = movie_df.iloc[top_tier_index_list].copy()

    top_tier_df['sim_score'] = simility_value_list

    return top_tier_df