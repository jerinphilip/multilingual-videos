import cv2            
import numpy as np
from .utils import get_iou
from collections import defaultdict

class Strategy:
    def __init__(self,
            group_bbox_detector, unit_bbox_detector,
            text_recognizer, translator, tgt_lang, inpainter, image_renderer
        ):
        self.group_bbox_detector = group_bbox_detector
        self.unit_bbox_detector = unit_bbox_detector
        self.translator = translator
        self.tgt_lang = tgt_lang
        self.inpainter = inpainter
        self.image_renderer = image_renderer
        self.text_recognizer = text_recognizer

    def __call__(self, image):
        return self.process(image)

    def process(self, image):
        _, group_boxes = self.group_bbox_detector.predict(image)
        _, unit_boxes = self.unit_bbox_detector.predict(image)
        texts = self.detect_texts(image, unit_boxes)
        collected = self._collect(group_boxes, unit_boxes, texts)
        image = self._inpaint(image, collected)
        translated = self._translate(image, collected)
        return translated

    def _collect(self, ctpn_bboxes, bboxes, texts):
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

        groups = group(ctpn_bboxes, data)
        for _group in groups:
            _data = sorted(groups[_group], key=h)
            groups[_group] = _data
        return groups


    def _translate(self, image, collected):
        def _prettify(text):
            tgt_text = ' '.join(text)
            tgt_text = tgt_text.replace(' ', '')
            tgt_text = tgt_text.replace('▁', ' ')
            return tgt_text


        for group_bbox, v in collected.items():
            # Fix this up.
            texts = [t['text'] for t in v]
            src_text = ' '.join(texts)
            translation = self.translator(src_text, tgt_lang=self.tgt_lang)
            tgt_text = translation[0]['tgt']
            text = _prettify(tgt_text)
            image = self.image_renderer(image, group_bbox, text)

        return image


    def _inpaint(self, image, collected):
        image_for_inpainting = image.copy()

        mask = np.zeros_like(image_for_inpainting)
        for group_bbox, v in collected.items():
            cv2.rectangle(
                    mask, 
                    (group_bbox.x, group_bbox.y), 
                    (group_bbox.X, group_bbox.Y), 
                    (255, 255, 255), 
                    thickness=cv2.FILLED
                )

            for datum in v:
                bbox = datum['bbox']
                cv2.rectangle(
                        mask, 
                        (bbox.x, bbox.y), (bbox.X, bbox.Y), 
                        (255, 255, 255), thickness=cv2.FILLED
                )

        image_for_inpainting = self.inpainter.predict(
                image_for_inpainting, 
                mask
        )

        return image_for_inpainting


    def detect_texts(self, image, unit_boxes):
        texts = []
        for i, bbox in enumerate(unit_boxes):
            cropped = image[bbox.y:bbox.Y, bbox.x:bbox.X, :]
            text = self.text_recognizer.predict(cropped)
            texts.append(text)
        return texts
