import numpy as np
import cv2
from argparse import ArgumentParser
from collections import namedtuple
from collections import defaultdict
from PIL import ImageFont, ImageDraw, Image

import sys
import os
from utils import get_iou

def collect(bboxes, ctpn_bboxes, texts, logfile=sys.stdout):
    data = [
        {'bbox': bbox, 'text': text}
        for bbox, text in zip(bboxes, texts)
    ]

    f = lambda d: (d["bbox"].x + d["bbox"].X, d["bbox"].y + d["bbox"].Y)
    g = lambda d: tuple(reversed(f(d)))
    h = lambda d: d["bbox"].x + d["bbox"].X

    def group(ctpn_bboxes, data):
        groups = defaultdict(list)
        for bbox in ctpn_bboxes:
            for datum in data:
                iarea = get_iou(bbox, datum['bbox'])
                if iarea > 0:
                    groups[bbox].append(datum)
        return groups

    translations = {}
    groups = group(ctpn_bboxes, data)
    for _group in groups:
        _data = sorted(groups[_group], key=h)
        texts = list(map(lambda x: x['text'], _data))
        string = ' '.join(texts)
        output = translator(string, tgt_lang='hi')
        for line in output:
            print('> ', line['src'], file=logfile)
            print('< ', line['tgt'], file=logfile)
            print('', file=logfile, flush=True)
            translations[_group] = output
        # for datum in _data:
        #     bbox = datum['bbox']
        #     text = datum['text']
        #     print('{}:{}, {}:{} \t {}'.format(bbox.y, bbox.Y, bbox.x, bbox.X, text))
    return groups, translations

def f(counter, frame):
    _image, ctpn_bboxes = ctpnw.predict(frame)
    image, bboxes = eastw.predict(frame)
    cv2.imwrite('/ssd_scratch/cvit/jerin/acl-temp/bbox-ctpn-{}.jpg'.format(counter), _image)
    cv2.imwrite('/ssd_scratch/cvit/jerin/acl-temp/bbox-east-{}.jpg'.format(counter), image)
    texts = []
    copy_f = frame.copy()
    mask = np.zeros_like(copy_f)
    logfile = open('/ssd_scratch/cvit/jerin/acl-temp/translations-{}.txt'.format(counter), 'w+')
    for i, bbox in enumerate(bboxes):
        # cropped = frame[x_start:x_end, y_start:y_end, :]
        cropped = frame[bbox.y:bbox.Y, bbox.x:bbox.X, :]
        # print(bbox, frame.shape, cropped is None)
        text = crnnw.predict(cropped)
        texts.append(text)
        box = [bbox.x, bbox.y, bbox.X, bbox.y, bbox.X, bbox.Y, bbox.x, bbox.Y]
        box = np.array(box)
        cv2.polylines(frame, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                                                          thickness=2)

        cv2.rectangle(mask, (bbox.x, bbox.y), (bbox.X, bbox.Y), (255, 255, 255), thickness=cv2.FILLED)

    # Use boxes from CTPN as well
    for i, bbox in enumerate(ctpn_bboxes):
        cv2.rectangle(mask, (bbox.x, bbox.y), (bbox.X, bbox.Y), (255, 255, 255), thickness=cv2.FILLED)

    
    inpainted_image = ginpw.predict(copy_f, mask)
    cv2.imwrite('/ssd_scratch/cvit/jerin/acl-temp/bbox-mask-{}.jpg'.format(counter), mask)
    cv2.imwrite('/ssd_scratch/cvit/jerin/acl-temp/bbox-inpainted-{}.jpg'.format(counter), inpainted_image)
    groups, translations = collect(bboxes, ctpn_bboxes, texts, logfile)

    working_copy = Image.fromarray(inpainted_image.copy())
    for key in groups:
        # print(translations[key][0])
        working_copy = write_text(working_copy, key, translations[key][0])

    back_to_cv = np.array(working_copy)
    cv2.imwrite('/ssd_scratch/cvit/jerin/acl-temp/bbox-ip+text-{}.jpg'.format(counter), back_to_cv)


    return frame

ROOT = os.path.dirname(__file__)
ROOT = os.path.join(ROOT, '..')
fonts_dir = os.path.join(ROOT, 'fonts')
font_path = os.path.join(fonts_dir, "NotoSerifDevanagari-Regular.ttf")
print(font_path)
font = ImageFont.truetype(font_path, 14, encoding="utf-8")

def write_text(image, bbox, translation):
    draw = ImageDraw.Draw(image)
    tgt_text = translation['tgt']
    tgt_text = tgt_text.replace(' ', '')
    tgt_text = tgt_text.replace('▁', ' ')
    draw.text((bbox.x, bbox.y), tgt_text, (0, 0, 0), font=font)
    return image
    pass

if __name__ == '__main__':
    CRNN = os.path.join(ROOT, 'crnn.pytorch')
    sys.path.insert(0, CRNN)

    CTPN = os.path.join(ROOT, 'text-detection-ctpn')
    sys.path.insert(0, CTPN)

    EAST = os.path.join(ROOT, 'EAST')
    sys.path.insert(0, EAST)

    GINP = os.path.join(ROOT,  'generative_inpainting')
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


    parser = ArgumentParser()
    parser.add_argument('--path', required=True, type=str)
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.path)
    counter = 1
    annotations = []
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
                frame = f(counter, frame)
                # print(frame)
                cv2.imwrite("/ssd_scratch/cvit/jerin/acl-temp/dec-{}.jpg".format(counter),
                        frame)
            else:
                break

        counter = counter + 1











