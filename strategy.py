import cv2            
from .utils import get_iou

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

    def process(self, image):
        group_boxes = self.group_bbox_detector.predict(image)
        unit_boxes = self.unit_bbox_detector.predict(image)
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
            # texts = list(map(lambda x: x['text'], _data))
            # string = ' '.join(texts)
            # output = translator(string, tgt_lang='hi')
            # for line in output:
            #     print('> ', line['src'], file=logfile)
            #     print('< ', line['tgt'], file=logfile)
            #     print('', file=logfile, flush=True)
            #     translations[_group] = output
            # # for datum in _data:
            # #     bbox = datum['bbox']
            # #     text = datum['text']
            # #     print('{}:{}, {}:{} \t {}'.format(bbox.y, bbox.Y, bbox.x, bbox.X, text))
        return groups


    def _translate(self, image, collected):
        def _prettify(text):
            tgt_text = ' '.join(texts)
            tgt_text = tgt_text.replace(' ', '')
            tgt_text = tgt_text.replace('▁', ' ')

        for group_bbox, v in collected.items():
            # Fix this up.
            text = _prettify(v['texts'])
            self.image_renderer(image, group_bbox, text)

        return image


    def _inpaint(self, image, collected):
        image_for_inpainting = image.copy()

        for group_bbox, v in collected.items():
            mask = np.zeros_like(image_for_inpainting)

            cv2.rectangle(
                    mask, 
                    (group_bbox.x, group_bbox.y), 
                    (group_bbox.X, group_bbox.Y), 
                    (255, 255, 255), 
                    thickness=cv2.FILLED
                )

            for bbox in v['units']:
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
