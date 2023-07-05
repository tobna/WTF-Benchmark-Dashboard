import re
from matplotlib import pyplot as plt
import numpy as np

"""
TAXONOMY: dict
    A dictionary containing information on different types of models for image classification.
    
    Keys:
        - Baseline (str): The name of the baseline category of models. It contains models like DeiT and ViT.
        - Baseline (Conv) (str): The name of the baseline category of models with convolutional layers. It contains models like ResNet50.
        - Low-Rank Attention (str): The name of the category of models with low-rank attention mechanisms. It contains models like Nystrom, Linformer, and XCiT.
        - Sparse Attention (str): The name of the category of models with sparse attention mechanisms. It contains models like SwinV2, Swin, Sinkhorn_Cait, HaloNet, Routing_ViT, and WaveViT.
        - Fixed Attention (str): The name of the category of models with learned pattern mechanisms. It contains models like Synthesizer_FD and Synthesizer_FR.
        - Kernel Attention (str): The name of the category of models with kernel attention mechanisms. It contains models like Performer, Linear_ViT, PolySA, and CoaT.
        - Hybrid Attention (str): The name of the category of models with hybrid attention mechanisms. It contains models like EfficientFormerV2 and CvT.
        - Non-Attention Shuffling (str): The name of the category of models with non-attention shuffling mechanisms. It contains models like FNet, GFNet, Mixer, and FocalNet.
        - Sequence Reduction (str): The name of the category of models with sequence reduction mechanisms. It contains models like EViT, Dynamic_ViT, Token_Learner, ToMe, AViT, STViT, and CaiT.
        - Modified MLP (str): The name of the category of models with MLP (Multi-Layer Perceptron) mechanisms. It contains models like Switch.
    
    Values:
        - marker (str): The marker style used to represent the category of models in a plot.
        - models (list): A list of models that belong to the corresponding category.
"""
TAXONOMY = {"Baseline":                 {'marker': 'o',         'models': ['ViT', 'DeiT'], 'colors': plt.cm.tab20.colors[6: 8], 'tax_color': '#d62728'},
            "Baseline (Conv)":          {'marker': 'D',         'models': ['ResNet50'], 'colors': [plt.cm.tab20b.colors[0]], 'tax_color': '#1f77b4'},
            "Low-Rank Attention":       {'marker': 'v',         'models': ['Nystrom', 'Linformer', 'XCiT'], 'colors': plt.cm.tab20b.colors[12:15], 'tax_color': '#8c564b'},
            "Sparse Attention":         {'marker': '8',         'models': ['SwinV2', 'Swin', 'Sinkhorn_Cait', 'HaloNet', 'Routing_ViT', 'WaveViT'], 'colors': [plt.cm.Greens_r(i) for i in np.linspace(0.1, 0.6, 6)], 'tax_color': '#2ca02c'},
            "Fixed Attention":          {'marker': '^',         'models': ['Synthesizer_FD', 'Synthesizer_FR'], 'colors': plt.cm.tab20b.colors[4:6], 'tax_color': '#7f7f7f'},
            "Kernel Attention":         {'marker': 'h',         'models': ['Performer', 'Linear_ViT', 'PolySA'], 'colors': plt.cm.tab20c.colors[1:4], 'tax_color': '#ff7f0e'},
            "Hybrid Attention":         {'marker': 'H',         'models': ['EfficientFormerV2', 'CvT', 'CoaT'], 'colors': plt.cm.tab20b.colors[8:11], 'tax_color': '#bcbd22'},
            "Non-Attention Shuffling":  {'marker': 'p',         'models': ['FNet', 'GFNet', 'Mixer', 'FocalNet'], 'colors': plt.cm.tab20b.colors[16:20], 'tax_color': '#9467bd'},
            "Sequence Reduction":       {'marker': (5, 0, 36), 'models': ['EViT', 'Dynamic_ViT', 'Token_Learner', 'ToMe', 'AViT', 'STViT', 'CaiT'], 'colors': [plt.cm.cividis_r(i) for i in np.linspace(0.1, 0.6, 7)], 'tax_color': '#e377c2'},
            "MLP Block":                 {'marker': 's',         'models': ['Switch'], 'colors': [plt.cm.Set1.colors[1]], 'tax_color': '#17becf'}}


_SIZES_MAP = {'tiny': 'Ti', 'small': 'S', 'base': 'B'}


def get_legend_order(name):
    tax_classes = list(TAXONOMY.keys())
    if name in tax_classes:
        return tax_classes.index(name)
    cls = get_taxonomy_class(name)
    if cls == 'none':
        return -1
    model_idx = _get_tax_idx(name)
    size_cls = 1 if '-ti' in name.lower() else (2 if '-s' in name.lower() else 3)
    cls_len = len(TAXONOMY[cls]['models'])
    # print(f"{name} -> class {cls}; model {model_cls}; size class {size_cls}; idx in taxonomy {model_idx}")
    return tax_classes.index(cls) + (model_idx + 1) / (cls_len + 2) + size_cls // 50


def _get_tax_idx(model):
    cls = get_taxonomy_class(model)
    if cls == 'none':
        return -1
    model_cls = get_model_class(model).replace(' ', '_')
    if model_cls in TAXONOMY[cls]['models']:
        return TAXONOMY[cls]['models'].index(model_cls)
    elif model_cls[:-4] in TAXONOMY[cls]['models']:
        return TAXONOMY[cls]['models'].index(model_cls[:-4])
    return len(TAXONOMY[cls]['models'])


def get_tax_color(name):
    if name in TAXONOMY.keys():
        # it's a taxonomy class, not a model.
        return TAXONOMY[name]['tax_color']
    tax_cls = get_taxonomy_class(name)
    # print(model_name, tax_cls, _get_model_tax_idx(model_name))
    if tax_cls == 'none':
        return 'black'
    return TAXONOMY[tax_cls]['colors'][_get_tax_idx(name)]


def get_edge_color(model_name, base_color='black'):
    return [base_color, 'white', '0.0', '0.5'][_get_tax_idx(model_name) % 4]


def get_model_name(model_name):
    """
    Returns a formatted model name based on the input model name.

    Parameters:
    -----------
    model_name : str
        Name of the model.

    Returns:
    --------
    str
        Formatted model name.

    Example:
    --------
    >>> get_model_name('vit_tiny_patch16')
    'ViT-Ti/16'
    """
    model_class = get_model_class(model_name)
    print_name = model_class
    patch_size = -1
    blocks = []
    for block in model_name.replace('-', '_').replace('/', '_').split('_'):
        m = re.search(r'\d+$', block)
        if m:
            num = str(int(m.group()))
            text = block[:-len(num)]
        else:
            num = -1

        if int(num) > 0 and text.lower() not in ['window', 'patch', 'p']:
            blocks.append(text)
            blocks.append(num)
        else:
            blocks.append(block)
    for block in blocks:
        # print(block)
        if block.lower() in model_class.lower():
            continue
        if block.lower() in ['vit', 'ls']:
            continue
        if block.isnumeric():
            print_name += f'-{int(block)}'
            continue
        if block.lower() in _SIZES_MAP.keys():
            print_name += f'-{_SIZES_MAP[block.lower()]}'
            continue
        if block.lower().startswith('p') and block[1:].isnumeric():
            patch_size = int(block[1:])
            continue
        if block.lower().startswith('patch') and block[5:].isnumeric():
            patch_size = int(block[5:])
            continue
        if block.lower().startswith('window'):
            print_name += f'-W{int(block[6:])}'
            continue
        print_name += f'-{block.capitalize()}'
    if block.isnumeric() and model_class.startswith('ViT') or model_class.startswith('Mixer'):
        print_name = print_name[:-len(f'-{int(block)}')]
        patch_size = int(block)
    if patch_size > 0:
        print_name += f'/{patch_size}'
    return print_name


def get_model_class(model_name):
    """
    Return the model class for a given model name.

    Parameters
    ----------
    model_name : str
        The name of the model to get the class for.

    Returns
    -------
    str
        The class name of the model.

    Notes
    -----
    The model class is determined by matching the given `model_name` against the `models` attribute
    of each class in the `TAXONOMY` dictionary. If a match is found, the corresponding class name
    is returned. If no match is found, the original `model_name` is returned.

    If the `model_name` is "nystrom", the class name "Nystrom_ViT" is returned. If the `model_name`
    is "switch", the class name "Switch_ViT" is returned.

    Examples
    --------
    >>> get_model_class("deit_small_patch16_224")
    "DeiT"

    >>> get_model_class("nystrom_vit_tiny_patch16_224")
    "Nystrom_ViT"
    """
    model_name = model_name.lower().replace(' ', '_')
    model_class = model_name
    for class_info in TAXONOMY.values():
        for arch in class_info['models']:
            if model_name.startswith(arch.lower()):
                model_class = arch
                break
    if model_class.lower() == 'nystrom':
        model_class = 'Nystrom_ViT'
    elif model_class.lower() == 'switch':
        model_class = 'Switch_ViT'
    model_class = model_class.replace('_', ' ')
    return model_class


def get_taxonomy_class(model_name):
    """
    Returns the taxonomy class name for a given model name.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    str
        Name of the taxonomy class the model belongs to.

    Notes
    -----
    If no match is found, the function returns "none".

    Examples
    --------
    >>> get_taxonomy_class('deit_small_ls_patch16')
    "Baseline"

    >>> get_taxonomy_class('Nystrom ViT-32-S/16')
    "Low-Rank Attention"
    """
    model_name = model_name.lower()
    model_name = model_name.replace(' ', '_')
    for class_name, class_info in TAXONOMY.items():
        if any(model_name.startswith(arch.lower()) for arch in class_info['models']):
            return class_name
    if model_name.endswith('_vit'):
        return get_taxonomy_class(model_name[:-4])
    if model_name.lower().startswith('efficientform'):
        return get_taxonomy_class('EfficientFormerV2')
    print(f"Could not find class for model '{model_name}'")
    return 'none'


def get_marker(model_name):
    """
    Given a model name, return the marker associated with its taxonomy class.

    Parameters
    ----------
    model_name : str
        The name of the model.

    Returns
    -------
    str
        The marker associated with the taxonomy class of the model. If the model
        does not belong to any taxonomy class, return 'o'.

    Examples
    --------
    >>> get_marker('vit_small_patch16_224')
    'o'
    >>> get_marker('Nystrom ViT-32-S/16')
    'v'
    >>> get_marker('unrecognized_model')
    'o'
    """
    tax_cls = get_taxonomy_class(model_name)
    if tax_cls == 'none':
        return 'o'
    return TAXONOMY[tax_cls]['marker']


