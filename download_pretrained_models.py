import argparse
import os
import requests
import re


"""
How to use:

download models:
    python main_download_pretrained_models.py --models "DPIR IRCNN" --model_dir "model_zoo"

"""


def download_pretrained_model(model_dir='./packages/drunet', model_name='drunet_gray.pth'):
    if os.path.exists(os.path.join(model_dir, model_name)):
        print(f'already exists, skip downloading [{model_name}]')
    else:
        os.makedirs(model_dir, exist_ok=True)
        if 'SwinIR' in model_name:
            url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(model_name)
        else:
            url = 'https://github.com/cszn/KAIR/releases/download/v1.0/{}'.format(model_name)
        r = requests.get(url, allow_redirects=True)
        print(f'downloading [{model_dir}/{model_name}] ...')
        open(os.path.join(model_dir, model_name), 'wb').write(r.content)
        print('done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',
                        type=lambda s: re.split(' |, ', s),
                        default = "drunet_gray.pth",
                        help='comma or space delimited list of characters, e.g., "DnCNN", "DnCNN BSRGAN.pth", "dncnn_15.pth dncnn_50.pth"')
    parser.add_argument('--model_dir', type=str, default='./packages/drunet', help='path of drunet_model')
    args = parser.parse_args()

    print(f'trying to download {args.models}')

    method_model_zoo = {'DPIR': ['drunet_gray.pth', 'drunet_color.pth'] }

    method_zoo = list(method_model_zoo.keys())
    model_zoo = []
    for b in list(method_model_zoo.values()):
        model_zoo += b

    if 'all' in args.models:
        for method in method_zoo:
            for model_name in method_model_zoo[method]:
                download_pretrained_model(args.model_dir, model_name)
    else:
        for method_model in args.models:
            if method_model in method_zoo:  # method, need for loop
                for model_name in method_model_zoo[method_model]:
                    if 'SwinIR' in model_name:
                        download_pretrained_model(os.path.join(args.model_dir, 'swinir'), model_name)
                    else:
                        download_pretrained_model(args.model_dir, model_name)
            elif method_model in model_zoo:  # model, do not need for loop
                if 'SwinIR' in method_model:
                    download_pretrained_model(os.path.join(args.model_dir, 'swinir'), method_model)
                else:
                    download_pretrained_model(args.model_dir, method_model)
            else:
                print(f'Do not find {method_model} from the pre-trained model zoo!')
