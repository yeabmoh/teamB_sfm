import plotly.express as px
import plotly.graph_objects as go
import os
import cv2
import numpy as np

def get_test_data(folder):
    """Input: folder
    Output: images, points (X, Y, Z), params (W, H)"""

    img_folder = os.path.join(folder, 'images')
    info_folder = os.path.join(folder, 'sparse')

    img_files = os.listdir(img_folder)
    images = [cv2.imread(os.path.join(img_folder, img_file)) for img_file in img_files]

    coordinates = []
    with open(os.path.join(info_folder, 'points3D.txt'), 'r') as points_file:
        for _ in range(3):
            next(points_file)
        for line in points_file:
            parts = line.strip().split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            coordinates.append((x, y, z))

    with open(os.path.join(info_folder, 'cameras.txt'), 'r') as cam_file:
        for _ in range(3):
            next(cam_file)
        for line in cam_file:
            parts = line.strip().split()
            params = [float(parts[i]) for i in range(2, 12)]
            
    return images, np.array(coordinates), params

def visualize_scene(pts, camera):
    scene_trace = go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',  # Change color or customize more as needed
            opacity=0.8
        ),
        name='Scene Points'
    )
    # Define the layout
    layout = go.Layout(
        title='3D Scene Reconstruction',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        margin=dict(l=0, r=0, b=0, t=0)  # Tight layout
    )
    # Create the figure combining both traces
    fig = go.Figure(data=[scene_trace], layout=layout)

    # add_plotly_camera(camera[0], camera[1], P, 1, fig)
    
    # Show the figure
    fig.show()