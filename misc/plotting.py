import cv2
import matplotlib
import matplotlib.pyplot as plt
import os

def plt_plot(img, save_path=None, title=None, cmap='viridis', additional_points=None, send_back=False):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Set the title of the plot
    ax.set_title(f"{title + ': ' if title is not None else ''}{img.shape}")
    
    # Display the image
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if cmap != 'gray' else img, cmap=cmap)
    
    # Plot additional points if provided
    if additional_points is not None:
        for p in additional_points:
            ax.plot(p[0], p[1], 'ro')
    
    # Adjust layout
    plt.tight_layout()   
    
    if save_path is not None:
        # Save the figure to the given path
        plt.savefig(save_path)
    elif send_back:
        # Return the figure object without showing or saving it
        return fig
    else:
        plt.show()
    plt.close(fig)

# def player_tracking_plots(players:dict, image):
#     color = 5
#     for player in players:
#         try:
#             cv2.circle(map_2d, (p.positions[timestamp]), 5, p.color, 3)
#             cv2.circle(map_2d, (p.positions[timestamp]), 7, (0, 0, 0), 1)
#             cv2.circle(map_2d_text, (p.positions[timestamp]), 13, p.color, -1)
#             cv2.circle(map_2d_text, (p.positions[timestamp]), 15, (0, 0, 0), 3)
#             text_size, _ = cv2.getTextSize(str(p.ID), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
#             text_origin = (p.positions[timestamp][0] - text_size[0] // 2,
#                             p.positions[timestamp][1] + text_size[1] // 2)
#             cv2.putText(map_2d_text, str(p.ID), text_origin,
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.75,
#                         (0, 0, 0), 2, cv2.LINE_AA)
#         except KeyError:
#             pass
