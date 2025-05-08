import torch
import yaml
import numpy as np
from torch.utils.data import Subset

#Save a KAN model to a file
def save_kan_model(model, path='model'):

    config = dict(
        grid_size=model.grid_size,
        spline_order=model.spline_order,
        layers_hidden=[layer.in_features for layer in model.layers] + [model.layers[-1].out_features],
        scale_noise=model.layers[0].scale_noise,
        scale_base=model.layers[0].scale_base,
        scale_spline=model.layers[0].scale_spline,
        grid_eps=model.layers[0].grid_eps,
        grid_range=model.layers[0].grid[0, :].tolist(),  # Assuming all layers share the same grid range
        base_activation=model.layers[0].base_activation.__class__.__name__,
    )

    with open(f'{path}_config.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    torch.save(model.state_dict(), f'{path}_state')

#Load a KAN model from a file
def load_kan_model(path='model'):
    with open(f'{path}_config.yml', 'r') as stream:
        config = yaml.safe_load(stream)

    state = torch.load(f'{path}_state')
    from efficient_kan.kan import KAN  # Import here to avoid circular imports
    model = KAN(
        layers_hidden=config['layers_hidden'],
        grid_size=config['grid_size'],
        spline_order=config['spline_order'],
        scale_noise=config['scale_noise'],
        scale_base=config['scale_base'],
        scale_spline=config['scale_spline'],
        base_activation=getattr(torch.nn, config['base_activation']),
        grid_eps=config['grid_eps'],
        grid_range=config['grid_range'],
    )

    model.load_state_dict(state)
    return model

#Evaluate the accuracy of a model
def evaluate(testloader, model, criterion):
    val_loss = 0
    val_accuracy = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.size(0), -1).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(testloader)
    val_accuracy /= len(testloader)
    return val_loss, val_accuracy

#get a subset of a given dataset
def data_subset(set, fraction=1):
    num_set = len(set)
    indices = np.random.permutation(num_set)[:int(fraction * num_set)]
    returnset = Subset(set, indices)
    return returnset
