#%%
import torch
import pandas as pd
import random

# generate a set of pseudo feature set (i.e. X)
def generate_pseudo_features_set(N, table_width=80, table_height=40):
    # define pocket direction vectors
    pocket_vectors = [{'x': -1, 'y': -1}, 
                     {'x': -1, 'y': 1},
                     {'x': 0, 'y': -1},
                     {'x': 0, 'y': 1},
                     {'x': 1, 'y': -1},
                     {'x': 1, 'y': 1}]
    
    # define pocket locations
    pocket_locations = []
    for i in range(3):
        for j in range(2):
            locations = {}
            locations['x'] = i * table_width / 2
            locations['y'] = j * table_height
            pocket_locations.append(locations)

    # generate pseudo features
    features_set = []
    for i in range(N):
        features = []
        
        # sample a white ball location
        white_ball_location = {}
        white_ball_location['x'] = torch.rand(1) * table_width
        white_ball_location['y'] = torch.rand(1) * table_height
        
        # sample a red ball location
        red_ball_location = {}
        red_ball_location['x'] = torch.rand(1) * table_width
        red_ball_location['y'] = torch.rand(1) * table_height

        # sample an available pocket
        idxs = list(range(len(pocket_locations)))
        random.shuffle(idxs)
        for idx in idxs:
            # calculate angle_1
            white_to_red = torch.Tensor([red_ball_location['x'] - white_ball_location['x'], 
                                         red_ball_location['y'] - white_ball_location['y']])
            red_to_pocket = torch.Tensor([pocket_locations[idx]['x'] - red_ball_location['x'], 
                                          pocket_locations[idx]['y'] - red_ball_location['y']])
            angle_1 = torch.acos(torch.dot(white_to_red, red_to_pocket) / (torch.norm(white_to_red) * torch.norm(red_to_pocket)))
            angle_1 = torch.rad2deg(angle_1)
            
            # if angle_1 >= 90, then pocket is not available
            if angle_1 >= 90:
                # print(angle_1, white_ball_location, red_ball_location, pocket_locations[idx])
                continue
            features.append(angle_1)
            
            # calculate angle_2
            red_to_pocket = torch.Tensor([pocket_locations[idx]['x'] - red_ball_location['x'], 
                                           pocket_locations[idx]['y'] - red_ball_location['y']])
            pocket_vector = torch.Tensor([pocket_vectors[idx]['x'], pocket_vectors[idx]['y']])
            angle_2 = torch.acos(torch.dot(red_to_pocket, pocket_vector) / (torch.norm(red_to_pocket) * torch.norm(pocket_vector)))
            angle_2 = torch.rad2deg(angle_2)
            features.append(angle_2)
            
            # calculate dist_1
            dist_1 = white_to_red.norm()
            features.append(dist_1)
            
            # calculate dist_2
            dist_2 = red_to_pocket.norm()
            features.append(dist_2)
            break
        
        features_set.append(features)
    return torch.Tensor(features_set)
        
# generate pseudo label (i.e. Y) from given features
def generate_pseudo_labels(X, threshold=0.5):    
    # calculate the pseudo influence of each input features
    y1 = (-torch.exp(X[:, 0]/17))
    y2 = (-torch.exp(X[:, 1]/30))
    y3 = (-torch.exp(X[:, 2]/25))
    y4 = (-torch.exp(X[:, 3]/25))
    
    def norm_y(y, noise=True, min=0):
        y = y - y.min()
        y = y * (1 - min) / y.max()
        y = y + min
        if noise:
            y = y + torch.normal(0, torch.ones(y.size()) * 0.05)
        return y
    
    # normalize each influence (i.e. y1, y2, y3, y4)
    y1 = norm_y(y1)
    y2 = norm_y(y2)
    y3 = norm_y(y3, min=0.3)
    y4 = norm_y(y4, min=0.3)
    
    # calculate the pseudo label
    label = (y1 * y2 * y3 * y4)
    if threshold:
        label = (label > threshold).float()
    
    return label

if __name__ == '__main__':
    # generate 500x pseudo shoot records
    X = generate_pseudo_features_set(N=300)
    Y = generate_pseudo_labels(X)
    
    dataset_dict = {'angle_1': X[:, 0], 'angle_2': X[:, 1], 'dist_1': X[:, 2], 'dist_2': X[:, 3], 'label': Y}
    dataset_df = pd.DataFrame(dataset_dict)
    dataset_df.to_csv('pseudo_dataset.csv', index=False)

#%%