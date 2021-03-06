import torch


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Parameters
    ----------
    filename : str
        Accept local filepath, URL, ``torchvision://xxx``.
    map_location : str, optional
        Same as :func:`torch.load`. Default: None.

    Returns
    -------
    checkpoint: {dict, OrderedDict}
        The loaded checkpoint. It can be either an OrderedDict storing model weights
        or a dict containing other information, which depends on the checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Parameters
    ----------
    model : Module
        Module to load checkpoint.
    filename : str
        Accept local filepath, URL, ``torchvision://xxx``.
    map_location : str
        Same as :func:`torch.load`.
    strict : bool
        Whether to allow different params for the model and checkpoint.
    logger : :mod:`logging.Logger`, optional
        The logger for error message.

    Returns
    -------
    checkpoint : dict or OrderedDict
        The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Parameters
    ----------
    module : Module
        Module that receives the state_dict.
    state_dict : OrderedDict
        Weights.
    strict : bool
        whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
    logger : :obj:`logging.Logger`, optional
        Logger to log the error message.
        If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        if prefix == 'backbone.':
            _all_prefix = [pre.split('.')[0] for pre in state_dict.keys()]
            if 'backbone' not in _all_prefix:
                prefix = '.'.join(prefix.split('.')[1:])
            pass

        # change first weight channel
        named_first_weight_list = [param for param in module.named_parameters()]
        if len(named_first_weight_list) > 0 and len(named_first_weight_list[0][1].shape) == 4:
            named_first_weight = named_first_weight_list[0]
            if named_first_weight[0] in state_dict and len(named_first_weight[1].shape) == 4:
                first_weight_channel = named_first_weight[1].shape[1]
                first_weight = state_dict[named_first_weight[0]]
                if first_weight_channel != first_weight.shape[1]:
                    first_weight = first_weight.mean(dim=1, keepdim=True)
                    first_weight = first_weight.repeat(1, first_weight_channel, 1, 1)
                    state_dict[named_first_weight[0]] = first_weight
                pass
            pass

        module._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if logger and len(err_msg) > 0:
        logger("\n".join(err_msg))

    return