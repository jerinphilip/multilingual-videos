import numpy as np
import cv2
from argparse import ArgumentParser
from collections import namedtuple
from collections import defaultdict
from PIL import ImageFont, ImageDraw, Image

import sys
import os
# from utils import get_iou
from .strategy import Strategy
from .renderer import ImageFontRenderer

def build_strategy():
    ROOT = os.path.dirname(__file__)
    ROOT = os.path.join(ROOT, '../../')
    fonts_dir = os.path.join(ROOT, 'fonts')
    font_path = os.path.join(fonts_dir, "NotoSerifDevanagari-Regular.ttf")
    
    image_renderer = ImageFontRenderer(font_path)

    CRNN = os.path.join(ROOT, 'crnn.pytorch')
    CTPN = os.path.join(ROOT, 'text-detection-ctpn')
    EAST = os.path.join(ROOT, 'EAST')
    GINP = os.path.join(ROOT,  'generative_inpainting')

    sys.path.insert(0, CRNN)
    sys.path.insert(0, CTPN)
    sys.path.insert(0, EAST)
    sys.path.insert(0, GINP)

    from ctpn import CTPNWrapper
    from east import EASTWrapper
    from crnn import CRNNWrapper
    from ginp.wrapper import GInpWrapper
    import ilmulti

    ctpnw = CTPNWrapper(
        checkpoint_path='/ssd_scratch/cvit/jerin/acl-workspace/checkpoints_mlt/'
    )

    eastw = EASTWrapper(
        checkpoint_path='/ssd_scratch/cvit/jerin/acl-workspace/east_icdar2015_resnet_v1_50_rbox/'
    )

    crnnw = CRNNWrapper(
        model_path = '/ssd_scratch/cvit/jerin/acl-workspace/crnn.pth',
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    )

    ginpw =  GInpWrapper(
        checkpoint_dir='/ssd_scratch/cvit/jerin/acl-workspace/release_imagenet_256',
    )

    translator = ilmulti.translator.pretrained.mm_all()

    strategy = Strategy(
            ctpnw, eastw, crnnw, translator, 'hi',
            ginpw, image_renderer
    )

    return strategy

# def write_text(image, bbox, translation):
#     draw = ImageDraw.Draw(image)
#     base = 14
#     tgt_text = translation['tgt']
#     tgt_text = tgt_text.replace(' ', '')
#     tgt_text = tgt_text.replace('▁', ' ')
# 
#     while True:
#         font = ImageFont.truetype(font_path, base, encoding="utf-8")
#         w, h = font.getsize(tgt_text)
#         if w > bbox.X - bbox.x or h > bbox.Y - bbox.y:
#             base = base - 1
#             break
#         base = base + 1
# 
#     font = ImageFont.truetype(font_path, base, encoding="utf-8")
#     draw.text((bbox.x, bbox.y), tgt_text, (0, 0, 0), font=font)
#     return image
#     pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True, type=str)
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.path)
    counter = 1
    annotations = []
    strategy = build_strategy()
    while(capture.isOpened()):
        return_code, frame = capture.read()
        fname = os.path.join('/ssd_scratch/cvit/jerin/acl-temp/', 
                    '{}.jpg'.format(counter)
                )
        # cv2.imwrite(fname, image)
        # print("New", counter, bboxes)
        SAMPLE = 120
        if counter % SAMPLE == 0:
            if frame is not None:
                frame = strategy.process(frame)
                # print(frame)
                cv2.imwrite("/ssd_scratch/cvit/jerin/acl-temp/dec-{}.jpg".format(counter),
                        frame)
            else:
                break

        counter = counter + 1











