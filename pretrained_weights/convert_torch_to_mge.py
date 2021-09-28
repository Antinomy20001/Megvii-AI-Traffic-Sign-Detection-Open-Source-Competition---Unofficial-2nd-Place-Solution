import megengine
import torch
import glob

def convert(model_name):
    print(f'Converting: {model_name}')
    weight = torch.load(f'{model_name}.pth', map_location='cpu')
    new_weight = {
        i.replace('backbone.', ''): weight['state_dict'][i].numpy()
        for i in weight['state_dict'].keys() if i.startswith('backbone.')
    }
    megengine.save(new_weight, f'{model_name}.pkl')
    print(f'Converting: {model_name} completed.')

if __name__ == '__main__':
    pths = [i.replace('.pth','') for i in glob.glob('./*.pth')]
    pkls = [i.replace('.pkl','') for i in glob.glob('./*.pkl')]
    
    wait_for_convert_models = (set(pths) - set(pkls))

    for model_name in wait_for_convert_models:
        convert(model_name)